import matplotlib

matplotlib.use("Agg")

import os
import json
import glob

import torch
import torch.nn as nn
import torchvision
import imageio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from PIL import Image
from skimage import measure
from torchvision import transforms
from torch.autograd import Variable
from shapely.geometry import Polygon
from shapely.algorithms.polylabel import polylabel
from matplotlib.backends.backend_agg import FigureCanvasAgg

import utils
import sem_rings
import data_utils
import refine_utils

from graph_utils import CropGraph
from models_img import ResGen
from find_corr import find_correspondence

interp_mode = transforms.InterpolationMode


class Refinery:
    def __init__(
        self,
        fp_id,
        ins_full,
        sem_full,
        img_full,
        generator,
        topo_net,
        pred=False,
        gt_load=False,
    ):
        self.fp_id = fp_id
        self.pred = pred
        self.ins_full = ins_full
        self.sem_full = sem_full
        self.img_full = img_full
        self.generator = generator
        self.topo_net = topo_net
        self.warnings = []
        self.areas_of_interest = []

        self.all_ins_edges = sem_rings.get_instance_edge_mapping(
            self.ins_full, self.sem_full, pad=False
        )
        self.all_sem_rings = sem_rings.get_sem_rings(
            self.all_ins_edges, self.ins_full, self.sem_full
        )

        if gt_load:
            self.gt_ins_full = np.load(
                utils.PREPROCESS_ROOT + "instance/%s.npy" % fp_id
            )
            self.gt_sem_full = np.load(
                utils.PREPROCESS_ROOT + "semantic/%s.npy" % fp_id
            )
            self.gt_sem_full[self.gt_sem_full == 24] = 7

            self.pred_to_gt = find_correspondence(
                self.ins_full, self.sem_full, self.gt_ins_full, self.gt_sem_full
            )

            self.gt_ins_edges = sem_rings.get_instance_edge_mapping(
                self.gt_ins_full, self.gt_sem_full, pad=False
            )
            self.gt_sem_rings = sem_rings.get_sem_rings(
                self.gt_ins_edges, self.gt_ins_full, self.gt_sem_full
            )

        self.refined_area = np.zeros_like(ins_full, dtype=bool)

        if self.pred:
            self.mark_symbols_for_removal()

    def mark_symbols_for_removal(self):
        self.symbol_needed = {}

        ins_ids = np.unique(self.ins_full)
        for ins_id in ins_ids:
            if data_utils.get_ins_sem(ins_id, self.ins_full, self.sem_full) != 7:
                continue

            needed = False
            ins_sem, ins_ring = self.all_sem_rings[ins_id]

            for (edge, n_sem, n_id) in ins_ring:
                if n_sem in [3, 4, 5]:
                    topo_vec = self.get_topo_vec(n_id)

                    if not topo_vec:
                        self.warnings.append(
                            "refinery, mark symbol, %d no topo vec" % n_id
                        )
                        continue

                    n_hist_dict = data_utils.get_four_way_hist(
                        self.ins_full, self.sem_full, self.all_sem_rings, n_id
                    )
                    n_side = self.edge_on_which_side(edge, n_hist_dict)
                    n_after_vec = np.round(topo_vec["after"]).astype(int)
                    n_after_hist = data_utils.restore_dict(n_after_vec)

                    if n_side and n_after_hist[n_side].max():
                        needed = True

            self.symbol_needed[ins_id] = needed

    def get_example_ids(self):
        example_ids = []
        ins_ids = np.unique(self.ins_full)

        for ins_id in ins_ids:
            ins_sem = data_utils.get_ins_sem(ins_id, self.ins_full, self.sem_full)

            if ins_sem not in [3, 4, 5]:
                continue

            if self.pred:
                # this instance is too small
                if np.sum(self.ins_full == ins_id) <= 50:
                    self.warnings.append("refinery, get IDs, %d too small" % ins_id)
                    continue

                topo_vec = self.get_topo_vec(ins_id)
                if not len(topo_vec.keys()):
                    self.warnings.append("refinery, get IDs, %d no topo vec" % ins_id)
                    continue

            example_ids.append(ins_id)

        return example_ids

    def get_graph(self, ins_id):
        ins_crop, sem_crop, bbox = self.crop_object(ins_id)
        mini, minj, maxi, maxj = bbox
        img_crop = self.img_full[mini:maxi, minj:maxj]

        if self.pred:
            topo_vec = self.get_topo_vec(ins_id)
        else:
            topo_vec = None

        # need to get separate instances
        old_ins_crop = ins_crop.copy()
        ins_mask = ins_crop == ins_id
        ins_crop = measure.label(sem_crop, background=-1, connectivity=1)

        ins_id = np.unique(ins_crop[ins_mask])
        assert len(ins_id) == 1
        ins_id = ins_id[0]

        # update "before" topology vector, now that we've resized
        # if self.pred:
        #   crop_ins_edges = sem_rings.get_instance_edge_mapping(ins_crop,
        #                                                        sem_crop,
        #                                                        pad=True)
        #   crop_sem_rings = sem_rings.get_sem_rings(crop_ins_edges,
        #                                            ins_crop, sem_crop)
        #   hist_dict = data_utils.get_four_way_hist(ins_crop, sem_crop,
        #                                            crop_sem_rings, ins_id)
        #   topo_vec['before'] = data_utils.get_label(hist_dict)

        # also check which symbols we can remove here
        remove_ids = []

        if self.pred:
            for _ins_id in np.unique(old_ins_crop):
                _ins_sem = data_utils.get_ins_sem(_ins_id, self.ins_full, self.sem_full)

                if (_ins_sem == 7) and (not self.symbol_needed[_ins_id]):
                    _ins_mask = old_ins_crop == _ins_id
                    _ins_id, _count = np.unique(ins_crop[_ins_mask], return_counts=True)
                    # assert len(_ins_id) == 1
                    remove_ids.append(_ins_id[np.argmax(_count)])

        # build the graph
        crop_graph = CropGraph(
            ins_crop, sem_crop, self.fp_id, ins_id, remove_ids, topo_vec, fix=self.pred
        )

        return crop_graph, img_crop

    def get_topo_vec(self, ins_id):
        assert data_utils.get_ins_sem(ins_id, self.ins_full, self.sem_full) in [3, 4, 5]

        if np.sum(self.ins_full == ins_id) <= 10:
            return {}

        _, ins_ring = self.all_sem_rings[ins_id]
        if not len(ins_ring):
            return {}

        hist_dict = data_utils.get_four_way_hist(
            self.ins_full, self.sem_full, self.all_sem_rings, ins_id
        )
        if not len(hist_dict.keys()):
            return {}

        ins_crop, sem_crop, bbox = self.crop_object(
            ins_id, crop_margin=16, min_side_len=256
        )

        # NOTE there was a very bad prediction, so we skip
        if not bbox:
            print("Veeeery bad prediction, %s %d" % (self.fp_id, ins_id))
            return {}

        mini, minj, maxi, maxj = bbox
        img_crop = self.img_full[mini:maxi, minj:maxj]

        side_len = 224
        ins_crop = utils.resize(ins_crop, (side_len, side_len))
        sem_crop = utils.resize(sem_crop, (side_len, side_len))
        img_crop = utils.resize(img_crop, (side_len, side_len), interp_mode.BILINEAR)

        img_crop = img_crop[np.newaxis, ...]
        ins_mask = (ins_crop == ins_id).astype(np.float32)[np.newaxis, ...]
        sem_crop = data_utils.to_onehot(sem_crop, num_classes=8)
        combined = np.concatenate([sem_crop, img_crop, ins_mask], axis=0)
        combined = torch.FloatTensor(combined)

        # CNN forward pass
        combined = combined.cuda().unsqueeze(0)
        topo_pred = self.topo_net(combined)
        topo_pred = torch.sigmoid(topo_pred)[0].detach().cpu().numpy()

        topo_vec = {
            "before": data_utils.get_label(hist_dict),
            "after": topo_pred.tolist(),
        }
        return topo_vec

    def get_corr_topo_vec(self, pred_id):
        gt_id = self.pred_to_gt[pred_id]
        if gt_id < 0:
            return None

        assert data_utils.get_ins_sem(gt_id, self.gt_ins_full, self.gt_sem_full) in [
            3,
            4,
            5,
        ]

        assert np.sum(self.gt_ins_full == gt_id) > 10

        _, ins_ring = self.gt_sem_rings[gt_id]
        if not len(ins_ring):
            return None

        hist_dict = data_utils.get_four_way_hist(
            self.gt_ins_full, self.gt_sem_full, self.gt_sem_rings, gt_id
        )
        if not len(hist_dict.keys()):
            return None

        return data_utils.get_label(hist_dict).tolist()

    def get_example(self, ins_id):
        ins_crop, sem_crop, bbox = self.crop_object(ins_id)
        mini, minj, maxi, maxj = bbox
        img_crop = self.img_full[mini:maxi, minj:maxj]

        if self.pred:
            topo_vec = self.get_topo_vec(ins_id)
        else:
            topo_vec = None

        # need to get separate instances
        old_ins_crop = ins_crop.copy()
        ins_mask = ins_crop == ins_id
        ins_crop = measure.label(sem_crop, background=-1, connectivity=1)

        ins_id = np.unique(ins_crop[ins_mask])
        assert len(ins_id) == 1
        ins_id = ins_id[0]

        # also check which symbols we can remove here
        remove_ids = []

        if self.pred:
            for _ins_id in np.unique(old_ins_crop):
                _ins_sem = data_utils.get_ins_sem(_ins_id, self.ins_full, self.sem_full)

                if (_ins_sem == 7) and (not self.symbol_needed[_ins_id]):
                    _ins_mask = old_ins_crop == _ins_id
                    _ins_id = np.unique(ins_crop[_ins_mask])
                    assert len(_ins_id) == 1
                    remove_ids.append(_ins_id[0])

        # build the graph
        crop_graph = CropGraph(
            ins_crop, sem_crop, self.fp_id, ins_id, remove_ids, topo_vec, fix=self.pred
        )

        for s_ins_crop, s_sem_crop, s_bbox in crop_graph.get_crops():
            s_mini, s_minj, s_maxi, s_maxj = s_bbox

            example = {
                "fp_id": self.fp_id,
                "fix_id": ins_id,
                "ins_crop": s_ins_crop,
                "sem_crop": s_sem_crop,
                "img_crop": img_crop[s_mini:s_maxi, s_minj:s_maxj],
                "crop_graph": crop_graph,
                "s_bbox": s_bbox,
            }
            yield example

    def get_examples(self):
        examples = []

        for ins_id in self.get_example_ids():
            for example in self.get_example(ins_id):
                examples.append(example)

        return examples

    def edge_on_which_side(self, edge, hist_dict):
        answer = ""
        a_0, b_0 = edge

        for side in ["top", "right", "bottom", "left"]:
            for (other_edge, _, _) in hist_dict[side + "_edge"]:
                a_1, b_1 = other_edge

                if (tuple(a_0) == tuple(a_1) and tuple(b_0) == tuple(b_1)) or (
                    tuple(a_0) == tuple(b_1) and tuple(b_0) == tuple(a_1)
                ):
                    assert not answer
                    answer = side

        return answer

    def crop_object(self, ins_id, crop_margin=16, min_side_len=128):
        ins_mask = self.ins_full == ins_id

        ii, jj = np.nonzero(ins_mask)
        side_len = max(
            ii.max() - ii.min() + 2 * crop_margin,
            jj.max() - jj.min() + 2 * crop_margin,
            min_side_len,
        )
        center_i = (ii.max() + ii.min()) // 2
        center_j = (jj.max() + jj.min()) // 2

        mini = max(center_i - side_len // 2, 0)
        minj = max(center_j - side_len // 2, 0)
        mini = min(mini, self.ins_full.shape[0] - side_len)
        minj = min(minj, self.ins_full.shape[1] - side_len)

        maxi = min(mini + side_len, self.ins_full.shape[0])
        maxj = min(minj + side_len, self.ins_full.shape[1])

        if (mini < 0) or (minj < 0):
            return None, None, None

        assert (mini >= 0) and (minj >= 0)
        assert (maxi <= self.ins_full.shape[0]) and (maxj <= self.ins_full.shape[1])

        ins_crop = self.ins_full[mini:maxi, minj:maxj]
        sem_crop = self.sem_full[mini:maxi, minj:maxj]

        return ins_crop, sem_crop, [mini, minj, maxi, maxj]

    def debug_vis(self, ins_id):
        ins_crop, sem_crop, bbox = self.crop_object(ins_id)

        plt.figure(figsize=(12, 8))
        plt.imshow(
            sem_crop / 7.0,
            cmap="nipy_spectral",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        plt.imshow(ins_crop, alpha=0)

        plt.axis("off")
        plt.tight_layout()
        plt.ion()
        plt.show()
        plt.pause(0.001)

    def debug_symbol_removal(self):
        raise Exception("Not implemented")

    def refine_objects(self):
        assert self.generator

        new_sem_full = self.sem_full.copy()
        example_ids = self.get_example_ids()

        for i, ins_id in enumerate(example_ids):
            print("\r%s (O): %d / %d" % (self.fp_id, i, len(example_ids)), end="")

            old_ins_crop, old_sem_crop, bbox = self.crop_object(ins_id)

            # NOTE there was a very bad prediction, so we skip
            if not bbox:
                print("Veeeery bad prediction, %s %d" % (self.fp_id, ins_id))
                continue

            old_ins_sem = data_utils.get_ins_sem(ins_id, old_ins_crop, old_sem_crop)

            if self.pred:
                topo_vec = self.get_topo_vec(ins_id)
            else:
                topo_vec = None

            # use our updated semantic segmentation
            mini, minj, maxi, maxj = bbox
            new_sem_crop = new_sem_full[mini:maxi, minj:maxj].copy()
            img_crop = self.img_full[mini:maxi, minj:maxj]

            # use reshaped crops for everything
            # old_sem_crop = utils.resize(old_sem_crop, [64,64])
            # old_ins_crop = utils.resize(old_ins_crop, [64,64])
            # new_sem_crop = utils.resize(new_sem_crop, [64,64])

            # need to get separate instances
            old_ins_id = ins_id
            ins_mask = old_ins_crop == ins_id
            new_ins_crop = measure.label(new_sem_crop, background=-1, connectivity=1)
            ins_id = data_utils.get_new_id(
                ins_mask, old_ins_sem, new_ins_crop, new_sem_crop
            )

            # occured once, this means that the original instance was somehow refined
            # out, so we don't worry about it anymore
            if ins_id == -1:
                continue

            # update "before" topology vector, maybe part of it is refined now
            crop_ins_edges = sem_rings.get_instance_edge_mapping(
                new_ins_crop, new_sem_crop, pad=True
            )
            crop_sem_rings = sem_rings.get_sem_rings(
                crop_ins_edges, new_ins_crop, new_sem_crop
            )
            hist_dict = data_utils.get_four_way_hist(
                new_ins_crop, new_sem_crop, crop_sem_rings, ins_id
            )
            # NOTE temporary again
            if not hist_dict:
                continue
            topo_vec["before"] = data_utils.get_label(hist_dict)

            # also check which symbols we can remove here
            remove_ids = []

            if self.pred:
                for _ins_id in np.unique(old_ins_crop):
                    _ins_sem = data_utils.get_ins_sem(
                        _ins_id, self.ins_full, self.sem_full
                    )

                    if (_ins_sem == 7) and (not self.symbol_needed[_ins_id]):
                        _ins_mask = old_ins_crop == _ins_id
                        _ins_id = data_utils.get_new_id(
                            _ins_mask, _ins_sem, new_ins_crop, new_sem_crop
                        )

                        if _ins_id >= 0:
                            remove_ids.append(_ins_id)
                        else:
                            self.warnings.append("refinery, no corr for something")

            # build the graph
            crop_graph = CropGraph(
                new_ins_crop,
                new_sem_crop,
                self.fp_id,
                ins_id,
                remove_ids,
                topo_vec,
                fix=True,
            )
            self.warnings.extend(crop_graph.warnings)

            need_refine = False
            for key in ["top", "right", "bottom", "left"]:
                if crop_graph.side_fixed[key]:
                    need_refine = True

            if need_refine:
                for crop_i, (s_ins_crop, s_sem_crop, s_bbox) in enumerate(
                    crop_graph.get_crops()
                ):
                    s_mini, s_minj, s_maxi, s_maxj = s_bbox

                    example = {
                        "fp_id": self.fp_id,
                        "fix_id": ins_id,
                        "ins_crop": s_ins_crop,
                        "sem_crop": s_sem_crop,
                        "img_crop": img_crop[s_mini:s_maxi, s_minj:s_maxj],
                        "crop_graph": crop_graph,
                        "s_bbox": s_bbox,
                    }

                    batch = data_utils.get_network_inputs(
                        example, hide_prob=[0, 0], shrink_prob=[0, 1]
                    )

                    # determine which area we need to fix constant
                    fix_margin = 8
                    fix_mask = np.zeros_like(self.refined_area[mini:maxi, minj:maxj])
                    fix_mask[:fix_margin, :] = True
                    fix_mask[-fix_margin:, :] = True
                    fix_mask[:, :fix_margin] = True
                    fix_mask[:, -fix_margin:] = True
                    fix_mask_64 = utils.resize(fix_mask, [64, 64])

                    # generator forward
                    refined_masks, step_vis, full_vis = self.iter_refine(
                        batch, fix_mask_64
                    )

                    # save full visualization for this window
                    new_mini = mini + s_mini
                    new_minj = minj + s_minj
                    new_maxi = mini + s_maxi
                    new_maxj = minj + s_maxj
                    new_bbox = [new_mini, new_minj, new_maxi, new_maxj]

                    if False:
                        fname = "./gifs_objects/%s_%03d_%d.gif" % (
                            self.fp_id,
                            old_ins_id,
                            crop_i,
                        )
                        self.save_step_vis(
                            fname, new_bbox, new_sem_full, batch, fix_mask, step_vis
                        )

                    # update our semantic mask
                    refined_sem_crop = refine_utils.convert_gen_masks(
                        refined_masks, batch["semantics"], new_bbox, fix_bits=True
                    )
                    new_sem_full[
                        new_mini:new_maxi, new_minj:new_maxj
                    ] = refined_sem_crop

            # save out where this is
            if True:
                before_topo_vec = topo_vec["before"].tolist()
                after_topo_vec = np.round(topo_vec["after"]).astype(int).tolist()
                gt_topo_vec = self.get_corr_topo_vec(old_ins_id)

                # fp_id, need_refine, before_match, after_match, bbox
                interest_str = "%s,%d,%d,%d,%d,%d,%d,%d" % (
                    self.fp_id,
                    need_refine,
                    before_topo_vec == gt_topo_vec,
                    after_topo_vec == gt_topo_vec,
                    mini,
                    minj,
                    maxi,
                    maxj,
                )
                self.areas_of_interest.append(interest_str)

            # save out qual vis of door-handle fixing
            if False:
                before_topo_vec = topo_vec["before"].tolist()
                after_topo_vec = np.round(topo_vec["after"]).astype(int).tolist()
                gt_topo_vec = self.get_corr_topo_vec(old_ins_id)

                fig, [ax1, ax2, ax3] = plt.subplots(figsize=(16, 9), ncols=3, dpi=150)

                ax1.imshow(
                    new_sem_crop / 7.0,
                    cmap="nipy_spectral",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                ax2.imshow(
                    new_sem_full[mini:maxi, minj:maxj] / 7.0,
                    cmap="nipy_spectral",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                ax3.imshow(
                    self.gt_sem_full[mini:maxi, minj:maxj] / 7.0,
                    cmap="nipy_spectral",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )

                ax1.set_title(
                    "Top: %d\n" % sum(before_topo_vec[:2])
                    + "Right: %d\n" % sum(before_topo_vec[2:4])
                    + "Bottom: %d\n" % sum(before_topo_vec[4:6])
                    + "Left: %d" % sum(before_topo_vec[6:8])
                )
                ax2.set_title(
                    "Top: %d\n" % sum(after_topo_vec[:2])
                    + "Right: %d\n" % sum(after_topo_vec[2:4])
                    + "Bottom: %d\n" % sum(after_topo_vec[4:6])
                    + "Left: %d" % sum(after_topo_vec[6:8])
                )

                if gt_topo_vec:
                    ax3.set_title(
                        "Top: %d\n" % sum(gt_topo_vec[:2])
                        + "Right: %d\n" % sum(gt_topo_vec[2:4])
                        + "Bottom: %d\n" % sum(gt_topo_vec[4:6])
                        + "Left: %d" % sum(gt_topo_vec[6:8])
                    )

                for a in crop_graph.G.nodes:
                    a_x, a_y = crop_graph.G.nodes[a]["centroid"]

                    if a not in crop_graph.ins_crop:
                        ax2.plot(a_x, a_y, "*c", markersize=10)
                    else:
                        ax2.plot(a_x, a_y, "oc", markersize=5)

                ax1.set_axis_off()
                ax2.set_axis_off()
                ax3.set_axis_off()

                plt.tight_layout()
                save_f = "./vis/%d_%d_%d/%s_%d.png" % (
                    need_refine,
                    before_topo_vec == gt_topo_vec,
                    after_topo_vec == gt_topo_vec,
                    self.fp_id,
                    ins_id,
                )
                os.makedirs(os.path.dirname(save_f), exist_ok=True)
                plt.savefig(save_f, bbox_inches="tight", pad_inches=0.1)
                plt.close()

        print("")

        # update our full semantic and instance masks
        self.sem_full = new_sem_full
        self.ins_full = measure.label(new_sem_full, background=-1, connectivity=1)

    def refine_windows(self):
        raise Exception("Not doing this")

        # precompute all the windows we will look at
        crop_window = 128
        crop_stride = 64

        windows = []
        h, w = self.ins_full.shape

        for i in range(0, h // crop_stride):
            mini = i * crop_stride
            maxi = mini + crop_window

            if mini >= h:
                continue
            if maxi > h:
                maxi = h
                mini = h - crop_window

            for j in range(0, w // crop_stride):
                minj = j * crop_stride
                maxj = minj + crop_window

                if minj >= w:
                    continue
                if maxj > w:
                    maxj = w
                    minj = w - crop_window

                windows.append([mini, minj, maxi, maxj])

        # refine window by window
        new_sem_full = self.sem_full.copy()

        for window_i, bbox in enumerate(windows):
            print("\r%s (W): %d / %d" % (self.fp_id, window_i, len(windows)), end="")
            mini, minj, maxi, maxj = bbox

            # get network inputs
            img_crop = self.img_full[mini:maxi, minj:maxj]
            old_sem_crop = new_sem_full[mini:maxi, minj:maxj].copy()
            old_ins_crop = measure.label(old_sem_crop, background=-1, connectivity=1)

            old_sem_crop = utils.resize(old_sem_crop, [64, 64])
            old_ins_crop = utils.resize(old_ins_crop, [64, 64])

            if len(np.unique(old_ins_crop)) == 1:
                continue

            batch = data_utils.get_network_inputs(
                old_ins_crop, old_sem_crop, img_crop, shrink_prob=[1, 1]
            )

            # determine which area we need to fix constant
            fix_margin = 8
            fix_mask = self.refined_area[mini:maxi, minj:maxj].copy()
            fix_mask[:fix_margin, :] = True
            fix_mask[-fix_margin:, :] = True
            fix_mask[:, :fix_margin] = True
            fix_mask[:, -fix_margin:] = True
            fix_mask_64 = utils.resize(fix_mask, [64, 64])

            # generator forward
            refined_masks, step_vis, full_vis = self.iter_refine(batch, fix_mask_64)

            # save full visualization for this window
            fname = "./gifs_objects/%s_%d.gif" % (self.fp_id, window_i)
            self.save_step_vis(fname, bbox, new_sem_full, batch, fix_mask, step_vis)

            full_vis.insert(0, np.zeros_like(full_vis[0]))
            full_fname = "./gifs_objects/%s_%d_full.gif" % (self.fp_id, window_i)
            imageio.mimsave(full_fname, full_vis, duration=0.5)

            # update our semantic mask
            refined_sem_crop = refine_utils.convert_gen_masks(
                refined_masks, batch["semantics"], bbox, fix_bits=True
            )
            new_sem_full[mini:maxi, minj:maxj] = refined_sem_crop

            # update the mask we use to keep things constant
            self.refined_area[mini:maxi, minj:maxj][~fix_mask] = True

        print("")

        # update our full semantic and instance masks
        self.sem_full = new_sem_full
        self.ins_full = measure.label(new_sem_full, background=-1, connectivity=1)

    def iter_refine(self, batch, fix_mask=None):
        Tensor = torch.cuda.FloatTensor
        fix_mask = torch.BoolTensor(fix_mask)

        given_masks = Variable(batch["given_masks"].type(Tensor)).unsqueeze(1)
        given_imgs = Variable(batch["given_imgs"].type(Tensor))
        ind_masks = Variable(batch["ind_masks"].type(Tensor)).unsqueeze(1)
        full_masks = Variable(batch["full_masks"].type(Tensor))
        semantics = Variable(batch["semantics"].type(Tensor))
        graph_edges = batch["graph_edges"]
        ignore_masks = batch["ignore_masks"]
        nd_to_sample = torch.zeros(len(given_masks), dtype=torch.int64).cuda()

        z_shape = [given_masks.shape[0], 128]

        # prep for when we need to fix a portion of the initial given masks
        initial_masks = batch["full_masks"].type(Tensor).clone()
        fix_mask = fix_mask.type(torch.cuda.BoolTensor).unsqueeze(0)
        fix_mask = fix_mask.repeat(len(initial_masks), 1, 1)

        step_vis = []
        full_vis = []
        with torch.no_grad():
            # run one step first, hiding everywhere
            z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
            gen_masks, G_hidden = self.generator(
                z, given_masks, ind_masks, semantics, graph_edges, None, given_imgs
            )
            gen_masks[fix_mask] = initial_masks[fix_mask]

            # save intermediate visualizations
            step_vis.append(gen_masks.detach().cpu().numpy())
            batch["gen_masks"] = gen_masks.detach()
            full_vis.append(utils.vis_input(batch))

            # run N steps of refinement
            prev_pred = None
            patience = 10

            for step_i in range(1, 50):
                given_masks = gen_masks.detach().unsqueeze(1)
                ind_masks = torch.zeros_like(ind_masks)

                # fix a portion of the given masks constant
                # before_mask = refine_utils.convert_gen_masks(given_masks.squeeze(), semantics)
                # after_mask = refine_utils.convert_gen_masks(given_masks.squeeze(), semantics)

                # fig, [ax1, ax2, ax3] = plt.subplots(ncols=3)
                # ax1.imshow(before_mask / 7., cmap='nipy_spectral',
                #            vmin=0., vmax=1., interpolation='nearest')
                # ax2.imshow(fix_mask[0,0].cpu().numpy(), cmap='gray')
                # ax3.imshow(after_mask / 7., cmap='nipy_spectral',
                #            vmin=0., vmax=1., interpolation='nearest')
                # plt.show()
                # plt.close()

                z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
                gen_masks, G_hidden = self.generator(
                    z, given_masks, ind_masks, semantics, graph_edges, None, given_imgs
                )
                gen_masks[fix_mask] = initial_masks[fix_mask]

                # save intermediate visualizations
                curr_pred = refine_utils.convert_gen_masks(gen_masks, semantics)
                step_vis.append(gen_masks.detach().cpu().numpy())

                batch["given_masks"] = given_masks[:, 0, :, :]
                batch["ind_masks"] = ind_masks[:, 0]
                batch["gen_masks"] = gen_masks
                full_vis.append(utils.vis_input(batch))

                if (prev_pred == curr_pred).all():
                    if patience:
                        patience -= 1
                    else:
                        break

                else:
                    prev_pred = curr_pred
                    patience = 10

        return gen_masks, step_vis, full_vis

    def save_step_vis(self, fname, bbox, sem_full, batch, fix_mask, step_vis):
        sem_init = refine_utils.convert_full_masks(
            batch["full_masks"], batch["semantics"]
        )
        step_final = refine_utils.convert_gen_masks(step_vis[-1], batch["semantics"])

        # NOTE temporary, get the GT floorplan crop
        mini, minj, maxi, maxj = bbox
        sem_gt_full = np.load(utils.PREPROCESS_ROOT + "semantic/%s.npy" % self.fp_id)
        sem_gt_full[sem_gt_full == 24] = 7
        sem_gt_crop = sem_gt_full[mini:maxi, minj:maxj]

        frames = []

        for step_i, step_masks in enumerate(step_vis):
            step_crop = refine_utils.convert_gen_masks(step_masks, batch["semantics"])

            fig = plt.figure(figsize=(20, 12), dpi=150)
            canvas = FigureCanvasAgg(fig)
            fig.patch.set_facecolor("lightgray")

            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(2, 3, 2)
            ax3 = plt.subplot(2, 3, 3)
            ax4 = plt.subplot(2, 3, 5)
            ax5 = plt.subplot(2, 3, 6)

            mini, minj, maxi, maxj = bbox
            ax1.imshow(
                sem_full / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            xx = [minj, maxj, maxj, minj, minj]
            yy = [mini, mini, maxi, maxi, mini]
            ax1.plot(xx, yy, "-c")

            ax2.imshow(
                sem_init / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax3.imshow(
                sem_gt_crop / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax4.imshow(
                step_crop / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax5.imshow(
                step_final / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )

            ax1.set_title("full floorplan")
            ax2.set_title("results")
            ax3.set_title("GT")
            ax4.set_title("step %d" % step_i)
            ax5.set_title("result")

            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.set_axis_off()

            plt.tight_layout()

            canvas.draw()
            buf = canvas.buffer_rgba()
            frames.append(np.asarray(buf))
            plt.close()

        imageio.mimsave(fname, frames, duration=0.2)

    def vis_graph_and_steps(self, batch, step_vis):
        # helper function to determine two instance's connection
        def is_touching(mask_a, mask_b):
            mask_ab = np.logical_or(mask_a > 0, mask_b > 0)
            mask_ab = measure.label(mask_ab, background=0, connectivity=1)

            if len(np.unique(mask_ab)) > 2:
                return False
            else:
                return True

        ins_crop = np.zeros([64, 64], dtype=np.int64)
        sem_crop = np.zeros([64, 64], dtype=np.int64)

        for ins_id, (full_mask, semantic) in enumerate(
            zip(batch["full_masks"], batch["semantics"])
        ):
            full_mask = full_mask.numpy() > 0.0
            semantic = semantic.numpy().argmax()

            ins_crop[full_mask] = ins_id
            sem_crop[full_mask] = semantic

        # find the centroids of each instance
        centroids = {}
        ins_edges = sem_rings.get_instance_edge_mapping(ins_crop, sem_crop)

        for ins_id in ins_edges.keys():
            if len(ins_edges[ins_id]):
                poly_coords = [(a[1], a[0]) for (a, b) in ins_edges[ins_id]]
                polygon = Polygon(poly_coords)
                centroid = polylabel(polygon)
                centroids[ins_id] = (centroid.x, centroid.y)
            else:
                centroids[ins_id] = (0, 0)

        # see how well the graph is kept
        empty_ids = []
        last_step = step_vis[-1]

        for ins_id, ins_mask in enumerate(last_step):
            if not (ins_mask > 0).max():
                empty_ids.append(ins_id)

        num_errors = 0
        for (a, c, b) in batch["graph_edges"].numpy():
            if c > 0:
                if (a in empty_ids) or (b in empty_ids):
                    num_errors += 1
                elif not is_touching(last_step[a], last_step[b]):
                    num_errors += 1
                else:
                    pass

            else:
                if (a in empty_ids) or (b in empty_ids):
                    pass
                elif is_touching(last_step[a], last_step[b]):
                    num_errors += 1

        # plot each step with the graph next to it
        frames = []

        for step_i, step_masks in enumerate(step_vis):
            fig, [ax1, ax2] = plt.subplots(figsize=(16, 9), ncols=2)
            canvas = FigureCanvasAgg(fig)

            step_crop = refine_utils.convert_gen_masks(step_masks, batch["semantics"])
            ax1.imshow(
                sem_crop / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax2.imshow(
                step_crop / 7.0,
                cmap="nipy_spectral",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )

            for (a, c, b) in batch["graph_edges"].numpy():
                a_x, a_y = centroids[a]
                b_x, b_y = centroids[b]

                black_effects = [pe.Stroke(linewidth=5, foreground="k"), pe.Normal()]

                if c > 0:
                    if (a in empty_ids) or (b in empty_ids):
                        ax1.plot(
                            [a_x, b_x], [a_y, b_y], "-oc", path_effects=black_effects
                        )
                    elif is_touching(step_masks[a], step_masks[b]):
                        ax1.plot(
                            [a_x, b_x], [a_y, b_y], "-og", path_effects=black_effects
                        )
                    else:
                        ax1.plot(
                            [a_x, b_x], [a_y, b_y], "-or", path_effects=black_effects
                        )

                else:
                    if (a in empty_ids) or (b in empty_ids):
                        pass
                    elif is_touching(step_masks[a], step_masks[b]):
                        ax1.plot(
                            [a_x, b_x], [a_y, b_y], "--or", path_effects=black_effects
                        )

            ax1.set_title("Input and graph")
            ax2.set_title("Step %d" % step_i)
            ax1.set_axis_off()
            ax2.set_axis_off()

            plt.tight_layout()
            canvas.draw()
            buf = canvas.buffer_rgba()
            frames.append(np.asarray(buf))
            plt.close()

        return frames, num_errors


def refine_pred(fp_id, generator, topo_net):
    img_full = (
        np.array(
            Image.open(utils.PREPROCESS_ROOT + "fp_img/%s.jpg" % fp_id),
            dtype=np.float32,
        )
        / 255.0
    )
    sem_full = np.load(utils.PREPROCESS_ROOT + "semantic_pred/%s.npy" % fp_id)
    ins_full = measure.label(sem_full, background=-1, connectivity=1)

    refiner = Refinery(
        fp_id,
        ins_full,
        sem_full,
        img_full,
        generator,
        topo_net,
        pred=True,
        gt_load=True,
    )

    refiner.refine_objects()

    save_f = utils.PREPROCESS_ROOT + "refined/%s.npy" % fp_id
    os.makedirs(os.path.dirname(save_f), exist_ok=True)
    np.save(save_f, refiner.sem_full, allow_pickle=False)

    with open(utils.PREPROCESS_ROOT + "refined/%s.txt" % fp_id, "w") as f:
        f.write("fp_id,need_refine,before_match,after_match,mini,minj,maxi,maxj\n")
        for line in refiner.areas_of_interest:
            f.write(line + "\n")


def refine_heuristic(fp_id, generator, topo_net):
    img_full = (
        np.array(
            Image.open(utils.PREPROCESS_ROOT + "fp_img/%s.jpg" % fp_id),
            dtype=np.float32,
        )
        / 255.0
    )
    sem_full = np.load(utils.PREPROCESS_ROOT + "heuristic/%s.npy" % fp_id)
    ins_full = measure.label(sem_full, background=-1, connectivity=1)

    refiner = Refinery(
        fp_id,
        ins_full,
        sem_full,
        img_full,
        generator,
        topo_net,
        pred=True,
        gt_load=True,
    )

    refiner.refine_objects()

    save_f = utils.PREPROCESS_ROOT + "heuristic_refined/%s.npy" % fp_id
    os.makedirs(os.path.dirname(save_f), exist_ok=True)
    np.save(save_f, refiner.sem_full, allow_pickle=False)

    with open(utils.PREPROCESS_ROOT + "heuristic_refined/%s.txt" % fp_id, "w") as f:
        f.write("fp_id,need_refine,before_match,after_match,mini,minj,maxi,maxj\n")
        for line in refiner.areas_of_interest:
            f.write(line + "\n")


if __name__ == "__main__":
    args = utils.parse_arguments()

    # load trained generator
    gan_hparams = utils.parse_config(args.gan_f)
    gan_root = "../ckpts/04_frame_correct/%s/" % gan_hparams["experiment_name"]

    generator = ResGen(gan_hparams)
    generator_ckpt_f = sorted(glob.glob(gan_root + "*.pth"))[-1]
    print("generator: ", generator_ckpt_f)
    generator_ckpt = torch.load(generator_ckpt_f)
    generator.load_state_dict(generator_ckpt["model_G"])
    generator.eval()
    generator.cuda()

    # load trained topology fixing network
    topo_hparams = utils.parse_config(args.topo_f)
    topo_root = "../ckpts/03_frame_detect/%s/" % topo_hparams["experiment_name"]

    topo_net = torchvision.models.resnet50(num_classes=8)
    topo_net.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
    topo_ckpt_f = topo_root + "best.pth"
    print("topo: ", topo_ckpt_f)
    topo_ckpt = torch.load(topo_ckpt_f)
    topo_net.load_state_dict(topo_ckpt["model"])
    topo_net.eval()
    topo_net.cuda()

    assert gan_hparams["test_id"] == topo_hparams["test_id"]

    split_json = utils.SPLITS_ROOT + "ids_%d.json" % gan_hparams["test_id"]
    with open(split_json, "r") as f:
        fp_ids = json.load(f)

    for fp_id in fp_ids:
        if args.method == "r":
            refine_pred(fp_id, generator, topo_net)
        elif args.method == "hr":
            refine_heuristic(fp_id, generator, topo_net)
        else:
            raise Exception("Unknown method")
