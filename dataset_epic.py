import numpy as np
import h5py
import json
import torch
import torch.utils.data as data
import os
import copy
import random
from util.utils import *


class EPICKitchensDataset(data.Dataset):
    """EPIC-Kitchens-100-verb dataset adapter for the MATR online-TAL pipeline.

    Mirrors THUMOS14Dataset's public interface and windowing/label-generation
    algorithm (dataset.py), but adapted for EPIC's data format:
      - one .npz feature file per video (key 'feats', shape (T, feat_dim)),
        single-stream (no separate flow array), instead of THUMOS's two
        whole-dataset pickles.
      - JSON annotations carry an explicit integer 'label_id', used directly
        as the class index, instead of THUMOS's sort-and-lookup label strings.
      - JSON subset values are 'training'/'validation', not THUMOS's literal
        'train'/'test', so subset filtering needs an explicit name mapping.
    """

    SUBSET_MAP = {'train': 'training', 'test': 'validation'}

    def __init__(self, args, subset):
        self.args = args
        self.subset = subset
        self.video_anno_path = args.video_anno
        self.video_len_path = args.video_len_file.format(self.subset)
        self.num_of_class = args.num_of_class
        self.segment_size = args.num_frame
        self.label_name = []
        self.gt_action = {}
        self.inputs = []
        self.inputs_all = []
        self.num_queries = args.num_queries
        self.rgb = args.rgb
        self.flow = args.flow
        self.video_feature_all_train = args.video_feature_all_train
        self.video_feature_all_test = args.video_feature_all_test
        self.batch = args.batch
        self.p_videos = args.p_videos

        self.detect_len = args.detect_len
        self.anti_len = args.anti_len
        self.max_memory_len = args.max_memory_len

        if not self.rgb:
            raise ValueError("EPICKitchensDataset requires args.rgb=True (its single 'feats' "
                              "array is consumed via the model's rgb-only input path).")
        if self.flow:
            raise ValueError("EPICKitchensDataset has no separate flow stream; set args.flow=False.")

        self.feature_dir = self.video_feature_all_train if self.subset == 'train' else self.video_feature_all_test

        self._getDatasetDict()
        self._validate_feature_files()
        self._loadFeaturelen()
        self._getedlen()
        self._devide_video_set()
        self._makeInputSeq()

        label_cache_path = args.ontal_label_file.format(
            self.subset, self.segment_size, self.num_queries,
            self.detect_len, self.anti_len, self.max_memory_len, self.p_videos)
        label_cache_path = label_cache_path.replace('.h5', f'_batch{self.batch}.h5')
        self._loadPropLabel(label_cache_path)

        keys = self.video_list
        self.feature_file = {}
        for key in keys:
            self.feature_file[key] = np.load(os.path.join(self.feature_dir, f"{key}.npz"))['feats']

        # padding: each video's feature length -> multiple of batch
        for key in keys:
            duration = len(self.feature_file[key])
            remaining = duration % self.batch
            padding_size = self.batch - remaining
            self.feature_file[key] = np.pad(
                self.feature_file[key], ((0, padding_size), (0, 0)), 'constant', constant_values=0)

        self.ori_feature_file = copy.deepcopy(self.feature_file)

        self._padfeatures()

    def _padfeatures(self):
        # parrallel setting
        self.feature_file = copy.deepcopy(self.ori_feature_file)

        max_total_len = max(self.total_video_len)
        for v_set in range(self.p_videos):
            video_name = self.total_video_set[v_set][-1]
            padding_size = max_total_len - self.total_video_len[v_set]
            self.feature_file[video_name] = np.pad(
                self.feature_file[video_name], ((0, padding_size), (0, 0)), 'constant', constant_values=0)

    def _getDatasetDict(self):
        if self.subset not in self.SUBSET_MAP:
            raise ValueError(f"EPICKitchensDataset subset must be one of {list(self.SUBSET_MAP)}, "
                              f"got {self.subset!r}")
        json_subset = self.SUBSET_MAP[self.subset]

        anno_database = load_json(self.video_anno_path)
        anno_database = anno_database['database']
        self.video_dict = {}
        for video_name, video_info in anno_database.items():
            if video_info['subset'] == json_subset:
                self.video_dict[video_name] = video_info

        self.label_name = [str(i) for i in range(self.num_of_class - 1)]

        video_list = list(self.video_dict.keys())
        self.video_list = sorted(video_list, key=lambda k: random.random())

        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def _validate_feature_files(self):
        missing = [v for v in self.video_list
                   if not os.path.exists(os.path.join(self.feature_dir, f"{v}.npz"))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} EPIC feature file(s) missing from {self.feature_dir}: {missing}")

    def _loadFeaturelen(self):
        if os.path.exists(self.video_len_path):
            self.video_len = load_json(self.video_len_path)
            return

        self.video_len = {}
        keys = self.video_list
        for key in keys:
            self.video_len[key] = len(np.load(os.path.join(self.feature_dir, f"{key}.npz"))['feats'])

        os.makedirs(self.video_len_path.split('/video_len')[0], exist_ok=True)
        outfile = open(self.video_len_path, "w")
        json.dump(self.video_len, outfile, indent=2)
        outfile.close()

    def _getedlen(self):
        self.second_to_frame = {}
        max_label_id_seen = -1
        for index in range(0, len(self.video_list)):
            video_name = self.video_list[index]

            video_info = self.video_dict[video_name]
            video_labels = video_info['annotations']
            gt_edlen = []

            second_to_frame = self.video_len[video_name] / float(video_info['duration'])
            self.second_to_frame[video_name] = second_to_frame
            for j in range(len(video_labels)):
                tmp_info = video_labels[j]
                tmp_start = tmp_info['segment'][0] * second_to_frame
                tmp_end = tmp_info['segment'][1] * second_to_frame
                tmp_label = int(tmp_info['label_id'])
                max_label_id_seen = max(max_label_id_seen, tmp_label)
                gt_edlen.append([tmp_end, tmp_end - tmp_start, tmp_label])

            gt_edlen = np.array(gt_edlen)
            self.gt_action[video_name] = gt_edlen

        if max_label_id_seen > self.num_of_class - 2:
            raise ValueError(
                f"EPIC annotations contain label_id={max_label_id_seen} but args.num_of_class="
                f"{self.num_of_class} only allows indices 0..{self.num_of_class - 2} for real "
                f"classes (index {self.num_of_class - 1} is reserved for background). Increase "
                f"args.num_of_class to at least {max_label_id_seen + 2}.")

    def _devide_video_set(self):
        self.padded_video_len = {}
        self.total_video_len = [0] * self.p_videos
        self.total_video_set = [[] for _ in range(self.p_videos)]
        for index in range(0, len(self.video_list)):
            video_name = self.video_list[index]
            duration = self.video_len[video_name]
            duration += (self.batch - duration % self.batch)

            self.padded_video_len[video_name] = duration
            min_index = self.total_video_len.index(min(self.total_video_len))
            self.total_video_len[min_index] += duration
            self.total_video_set[min_index].append(video_name)

    def _makeInputSeq(self):
        self.inputs_all = [[] for _ in range(self.p_videos)]
        max_total_len = max(self.total_video_len)
        for v_set in range(self.p_videos):
            for video_name in self.total_video_set[v_set]:
                duration = self.padded_video_len[video_name]
                if video_name == self.total_video_set[v_set][-1]:
                    duration += max_total_len - self.total_video_len[v_set]
                for i in range(1, duration + 1):
                    st = i - self.segment_size
                    ed = i
                    self.inputs_all[v_set].append([video_name, st, ed])

        self.inputs = self.inputs_all.copy()
        print("%s subset seg numbers: %d" % (self.subset, len(self.inputs_all[0]) * self.p_videos))

    def _loadPropLabel(self, filename):
        if os.path.exists(filename):
            prop_label_file = h5py.File(filename, 'r')
            self.cls_label = np.array(prop_label_file['cls_label'][:])
            self.reg_label = np.array(prop_label_file['reg_label'][:])
            self.flag_label = np.array(prop_label_file['flag_label'][:])
            self.stcls_label = np.array(prop_label_file['stcls_label'][:])
            prop_label_file.close()
            return

        cls_label = [[] for _ in range(self.p_videos)]
        reg_label = [[] for _ in range(self.p_videos)]
        flag_label = [[] for _ in range(self.p_videos)]
        stcls_label = [[] for _ in range(self.p_videos)]
        max_inst_cnt = 0

        for v_set in range(self.p_videos):
            inputs_all = self.inputs_all[v_set]
            for i in range(0, len(inputs_all)):
                video_name = inputs_all[i][0]
                ed = inputs_all[i][2]
                target_boxes = self.gt_action[video_name]
                tmp_cls_label = []
                tmp_reg_label = []
                tmp_stcls_label = []
                tmp_target_boxes = []
                y_box = [ed - 1, self.segment_size]
                flag_gt = 0

                target_sel = [y_box[0] - self.detect_len + 1, y_box[0] + self.anti_len]
                target_sel_edlen = [target_sel[1], target_sel[1] - target_sel[0]]
                for target_box in target_boxes:
                    if frame_include(target_box[0], target_sel_edlen):
                        tmp_target_boxes.append(target_box)
                    if calc_iou(y_box, target_box) > 0.01:
                        flag_gt = 1

                if len(tmp_target_boxes) != 0:
                    for tmp_target_box in tmp_target_boxes:
                        v1 = np.zeros(self.num_of_class)
                        st_cls = np.zeros(self.max_memory_len + 2)
                        reg = np.zeros(2)

                        cls = int(tmp_target_box[2])
                        v1[cls] = 1
                        ed = tmp_target_box[0]
                        st = tmp_target_box[0] - tmp_target_box[1]
                        st_off = y_box[0] - st
                        ed_off = (y_box[0] - ed) / self.segment_size

                        st_sel = int(min(max(0, st_off // self.segment_size), self.max_memory_len + 1))
                        st_cls[st_sel] = 1
                        reg[0] = st_off / self.segment_size - st_sel
                        reg[1] = ed_off
                        tmp_cls_label.append(v1)
                        tmp_reg_label.append(reg)
                        tmp_stcls_label.append(st_cls)

                if len(tmp_target_boxes) != self.num_queries:
                    for _ in range(self.num_queries - len(tmp_target_boxes)):
                        v1 = np.zeros(self.num_of_class)
                        st_cls = np.zeros(self.max_memory_len + 2)
                        reg = np.zeros(2)
                        v1[-1] = 1
                        reg[-1] = -1e4
                        tmp_cls_label.append(v1)
                        tmp_reg_label.append(reg)
                        tmp_stcls_label.append(st_cls)

                if max_inst_cnt < len(tmp_target_boxes):
                    max_inst_cnt = len(tmp_target_boxes)

                tmp_cls_label = np.array(tmp_cls_label)
                tmp_reg_label = np.array(tmp_reg_label)
                tmp_stcls_label = np.array(tmp_stcls_label)

                cls_label[v_set].append(tmp_cls_label)
                reg_label[v_set].append(tmp_reg_label)
                flag_label[v_set].append(flag_gt)
                stcls_label[v_set].append(tmp_stcls_label)

        self.cls_label = np.stack(cls_label, axis=0)
        self.reg_label = np.stack(reg_label, axis=0)
        self.flag_label = np.stack(flag_label, axis=0)
        self.stcls_label = np.stack(stcls_label, axis=0)

        outfile = h5py.File(filename, 'w')
        dset_cls = outfile.create_dataset('/cls_label', self.cls_label.shape, maxshape=self.cls_label.shape, chunks=True, dtype=np.float32)
        dset_cls[:, :] = self.cls_label[:, :]
        dset_reg = outfile.create_dataset('/reg_label', self.reg_label.shape, maxshape=self.reg_label.shape, chunks=True, dtype=np.float32)
        dset_reg[:, :] = self.reg_label[:, :]
        dset_flag = outfile.create_dataset('/flag_label', self.flag_label.shape, maxshape=self.flag_label.shape, chunks=True, dtype=np.float32)
        dset_flag[:, ] = self.flag_label[:, ]
        dset_stcls = outfile.create_dataset('/stcls_label', self.stcls_label.shape, maxshape=self.stcls_label.shape, chunks=True, dtype=np.float32)
        dset_stcls[:, :] = self.stcls_label[:, :]

        outfile.close()

        return

    def __getitem__(self, index):
        gt = []
        info = []
        feature = []

        for v_set in range(self.p_videos):
            video_name, st, ed = self.inputs[v_set][index]
            if st >= 0:
                _feature = self._get_base_data(video_name, st, ed)
            else:
                _feature = self._get_base_data(video_name, 0, ed)
                padfunc2d = torch.nn.ConstantPad2d((0, 0, -st, 0), 0)
                _feature = padfunc2d(_feature)
            cls_label = torch.Tensor(self.cls_label[v_set][index])
            reg_label = torch.Tensor(self.reg_label[v_set][index])
            stcls_label = torch.Tensor(self.stcls_label[v_set][index])

            _gt = {
                'cls_label': cls_label,
                'reg_label': reg_label,
                'stcls_label': stcls_label
            }

            _info = {
                "video_name": video_name,
                "current_frame": ed - 1,
                "ed": ed,
                "st": st,
                "duration": self.video_len[video_name],
                "video_time": float(self.video_dict[video_name]["duration"]),
                "frame_to_time": float(self.video_dict[video_name]["duration"]) / self.video_len[video_name],
                "segment_flag": self.flag_label[v_set][index]
            }

            gt.append(_gt)
            info.append(_info)
            feature.append(_feature)
        return feature, gt, info

    def _get_base_data(self, video_name, st, ed):
        feature = self.feature_file[video_name][st:ed, :]
        feature = torch.from_numpy(np.array(feature))
        return feature

    def __len__(self):
        return len(self.inputs[0])

    def reset_sample(self):
        self.inputs = self.inputs_all.copy()

    def select_sample(self, idx):
        inputs = [self.inputs_all[0][i] for i in idx]
        self.inputs = inputs.copy()
        return
