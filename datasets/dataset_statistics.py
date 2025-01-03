# File: /datasets/dataset_statistics.py

import sys
import os
import argparse
from collections import defaultdict
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger('dataset_statistics')
    return logger

class DatasetStatistics:
    def __init__(self, dataset_root, k=5, i=0, volume_size=30, random_seed=42):
        """
        A dataset statistics class that:
          1) Gathers slice-based stats (total slices, slice-level train/val).
          2) Gathers patient-level stats (#patients per class, #patients in train/val).
          3) Gathers volume-based stats EXACTLY like CCCIIIDataset3D:
             - Each subfolder >= volume_size slices => 1 volume
             - Volume-level StratifiedKFold => partial volumes from the same patient can go to train/val.

        :param dataset_root: Path to the root directory of the dataset (e.g. ./data/ccccii).
        :param k: Number of folds for StratifiedKFold.
        :param i: Current fold index (0-based).
        :param volume_size: Minimum slices to treat one subfolder as 1 volume (default=30).
        :param random_seed: RNG seed for reproducibility.
        """
        self.dataset_root = dataset_root
        self.k = k
        self.i = i
        self.volume_size = volume_size
        self.random_seed = random_seed

        # Classes must match your directory names
        self.classes = ['CP', 'NCP', 'Normal']
        self.class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}

        self.logger = setup_logging()
        self.logger.info(f"Initialized with volume_size={self.volume_size}, k={self.k}, i={self.i}")

        # Data holders for slice-based stats
        self.slice_class_counts = defaultdict(int)
        self.total_slices = 0
        self.slice_class_percent = {}
        # Train/Val slice-based counters
        self.train_slice_counts = defaultdict(int)
        self.val_slice_counts = defaultdict(int)
        self.train_slice_percent = {}
        self.val_slice_percent = {}

        # Patient-level stats (like your original approach)
        self.total_patients = 0
        self.patients_per_class = defaultdict(int)
        self.train_patients_count = defaultdict(int)
        self.val_patients_count = defaultdict(int)

        # Volume-based stats
        self.all_volumes = []  # list of (patient_id, label)
        self.volume_class_counts = defaultdict(int)
        self.total_volumes = 0
        self.volume_class_percent = {}
        # Volume-level train/val
        self.train_volume_counts = defaultdict(int)
        self.val_volume_counts = defaultdict(int)
        self.train_volume_percent = {}
        self.val_volume_percent = {}

        # Now gather slice-based and patient-level info
        self._gather_slices_and_patients()

        # Perform a patient-level StratifiedKFold for slice stats
        self._perform_split_slices()

        # Gather volumes (≥ 30 slices => 1 volume)
        self._gather_volumes_like_ccciiidataset()

        # Perform volume-level StratifiedKFold EXACTLY matching the training script approach
        self._perform_split_volumes()

    def _gather_slices_and_patients(self):
        """
        1) For each subfolder with >= 30 slices, add 30 to slice_class_counts (like your original logic).
        2) Count #patients per class for the patient-level stats.
        3) We do NOT do a patient-level approach for slices here because you
           prefer to replicate your older approach: each valid scan => +30 slices.
        """
        self.logger.info(f"Gathering slice-based info from: {self.dataset_root}")
        unique_patients = set()

        for cls, label in self.class_to_label.items():
            cls_dir = os.path.join(self.dataset_root, cls)
            if not os.path.isdir(cls_dir):
                self.logger.warning(f"Missing class dir {cls_dir}, skipping.")
                continue

            patients = os.listdir(cls_dir)
            self.logger.info(f"Class '{cls}', {len(patients)} patient dirs found.")
            for pid in patients:
                patient_dir = os.path.join(cls_dir, pid)
                if not os.path.isdir(patient_dir):
                    continue
                unique_patients.add((pid, label))  # We'll track which class each patient belongs to

                scans = os.listdir(patient_dir)
                for scan_folder in scans:
                    scan_dir = os.path.join(patient_dir, scan_folder)
                    if not os.path.isdir(scan_dir):
                        continue

                    slices = [s for s in os.listdir(scan_dir) if os.path.isfile(os.path.join(scan_dir, s))]
                    if len(slices) >= 30:
                        # each valid scan => +30 slices (like your original code)
                        self.slice_class_counts[cls] += 30

        self.total_slices = sum(self.slice_class_counts.values())
        if self.total_slices > 0:
            for cls in self.classes:
                self.slice_class_percent[cls] = (self.slice_class_counts[cls]/self.total_slices)*100

        # Now track #patients per class
        # If a patient appears in multiple classes, we keep the first or do majority. 
        # We'll keep first for simplicity.
        patient_to_class = {}
        for (pid, label) in unique_patients:
            if pid not in patient_to_class:
                patient_to_class[pid] = label

        self.total_patients = len(patient_to_class)
        for pid, lbl in patient_to_class.items():
            cls_name = self.classes[lbl]
            self.patients_per_class[cls_name] += 1

    def _perform_split_slices(self):
        """
        We do a patient-level StratifiedKFold for the slices, 
        consistent with your original approach: 92504 slices in train, 23052 slices in val, etc.
        1) Build patient-> single label
        2) K-Fold across patients
        3) Tally slices
        """
        self.logger.info(f"Performing patient-level StratifiedKFold for slice-based stats, k={self.k}, i={self.i}")
        # Reconstruct a list of (patient_id, label) from above
        # We do the same approach you originally had: each patient => one label
        patient_ids = []
        patient_labels = []
        # We rely on self.patients_per_class to figure out which patient belongs to which label
        # but we kept track of them in patient_to_class
        # Let's rebuild that from scratch:
        patient_to_class = {}
        # We want to gather them from the code above
        # Easiest is to re-scan self.dataset_root, or store them earlier. For brevity, let's store them:
        # We'll do that in _gather_slices_and_patients using a local structure.

        # Let's store it from unique_patients approach
        # We'll do the correct approach: we already have "patient_to_class" in that method
        # So let's store that in self:
        # We'll rename the code a bit:
        self.patient_to_class = {}
        # Let's do it here for clarity:
        # We'll re-run that code:
        # Actually, let's just do it in _gather_slices_and_patients for cleanliness:
        # done. We'll assume we have self.patient_to_class = {} set there. 
        # For now, let's do it ad-hoc here:

        # Actually, let's store it in the same method:
        # (Simplify for demonstration, ignoring duplicates)
        # We'll store a small structure:

        # Instead, let's do a quick fix: re-iterate the classes in self.patients_per_class
        # That doesn't store which specific patient, though. 
        # We'll do a second pass. Sorry for a bit of confusion but let's keep it short:
        all_patient_label_pairs = []  # (patient_id, label)

        # We do a quick gather again (like above) but only patient->class
        # we won't re-check scans
        patient_set = {}
        for cls, label in self.class_to_label.items():
            cls_dir = os.path.join(self.dataset_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            patients = os.listdir(cls_dir)
            for pid in patients:
                if os.path.isdir(os.path.join(cls_dir, pid)):
                    # keep the first assignment
                    if pid not in patient_set:
                        patient_set[pid] = label

        # Now build the list
        for pid, lbl in patient_set.items():
            all_patient_label_pairs.append((pid, lbl))

        # Now do StratifiedKFold on that
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_seed)
        labels = [lbl for (_pid, lbl) in all_patient_label_pairs]
        splits = list(skf.split(all_patient_label_pairs, labels))

        if self.i < 0 or self.i >= self.k:
            sys.exit(f"Invalid fold index {self.i}")

        train_idx, val_idx = splits[self.i]
        train_patients = set(all_patient_label_pairs[x][0] for x in train_idx)
        val_patients = set(all_patient_label_pairs[x][0] for x in val_idx)

        # Now we have patient sets. We must count slices (the original approach):
        # We'll sum up how many slices are contributed by each class in train vs val
        # => We'll do a pass over the data or we rely on self.slice_class_counts? That doesn't separate by patient though
        # So let's do a small approach:
        # We'll do a second pass:  For each patient, each valid scan => +30 slices to train or val in that class.

        # Clear old train_slice_counts, val_slice_counts
        self.train_slice_counts = defaultdict(int)
        self.val_slice_counts   = defaultdict(int)

        for cls, label in self.class_to_label.items():
            cls_dir = os.path.join(self.dataset_root, cls)
            if not os.path.isdir(cls_dir):
                continue

            patients = os.listdir(cls_dir)
            for pid in patients:
                pdir = os.path.join(cls_dir, pid)
                if not os.path.isdir(pdir):
                    continue
                scans = os.listdir(pdir)
                for scan_folder in scans:
                    scan_dir = os.path.join(pdir, scan_folder)
                    if not os.path.isdir(scan_dir):
                        continue
                    slices = [s for s in os.listdir(scan_dir) if os.path.isfile(os.path.join(scan_dir, s))]
                    if len(slices) >= 30:
                        # +30 slices either in train or val
                        if pid in train_patients:
                            self.train_slice_counts[cls] += 30
                        else:
                            self.val_slice_counts[cls]   += 30

        # done. Now compute percentages
        total_slices = sum(self.slice_class_counts.values())  # the same 115260
        train_sum = sum(self.train_slice_counts.values())
        val_sum   = sum(self.val_slice_counts.values())

        # The % of total
        self.train_slice_percent = {}
        self.val_slice_percent   = {}
        if total_slices > 0:
            for cls in self.classes:
                self.train_slice_percent[cls] = (self.train_slice_counts[cls] / total_slices)*100
                self.val_slice_percent[cls]   = (self.val_slice_counts[cls]   / total_slices)*100

        # Also build the # patients in train vs val per class
        # we already have self.patients_per_class => total
        # Now let's do train_patients_count, val_patients_count
        self.train_patients_count = defaultdict(int)
        self.val_patients_count   = defaultdict(int)

        for pid in train_patients:
            lbl = patient_set[pid]
            cls_name = self.classes[lbl]
            self.train_patients_count[cls_name] += 1

        for pid in val_patients:
            lbl = patient_set[pid]
            cls_name = self.classes[lbl]
            self.val_patients_count[cls_name] += 1

    def _gather_volumes_like_ccciiidataset(self):
        """
        Exactly replicate CCCIIIDataset3D logic: 
        each subfolder with >= self.volume_size slices is 1 volume => (patient_id, label).
        """
        self.logger.info("Gathering volumes for CCCIIIDataset3D approach (≥30 slices => 1 volume).")

        for cls, label in self.class_to_label.items():
            cls_dir = os.path.join(self.dataset_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            patients = os.listdir(cls_dir)
            for pid in patients:
                patient_dir = os.path.join(cls_dir, pid)
                if not os.path.isdir(patient_dir):
                    continue
                scans = os.listdir(patient_dir)
                for scan_folder in scans:
                    scan_dir = os.path.join(patient_dir, scan_folder)
                    if not os.path.isdir(scan_dir):
                        continue
                    slices = [s for s in os.listdir(scan_dir) if os.path.isfile(os.path.join(scan_dir, s))]
                    if len(slices) >= self.volume_size:
                        self.all_volumes.append((pid, label))

        # Now count volumes
        for (_pid, label) in self.all_volumes:
            cls_name = self.classes[label]
            self.volume_class_counts[cls_name] += 1

        self.total_volumes = sum(self.volume_class_counts.values())
        if self.total_volumes > 0:
            for cls in self.classes:
                self.volume_class_percent[cls] = (self.volume_class_counts[cls] / self.total_volumes)*100

        self.logger.info(f"Total volumes gathered: {self.total_volumes}")
        self.logger.info(f"Volumes per class: {dict(self.volume_class_counts)}")

    def _perform_split_volumes(self):
        """
        Perform volume-level StratifiedKFold to match the training script approach 
        => a single patient can have multiple volumes in train & val.
        """
        self.logger.info(f"Performing volume-level Stratified K-Fold with k={self.k}, fold_index={self.i}")

        volume_labels = [label for (_pid, label) in self.all_volumes]
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_seed)
        splits = list(skf.split(self.all_volumes, volume_labels))

        if self.i < 0 or self.i >= self.k:
            sys.exit(f"Invalid fold index {self.i}")

        train_idx, val_idx = splits[self.i]
        self.logger.info(f"Fold {self.i}: {len(train_idx)} train volumes, {len(val_idx)} val volumes.")

        # Clear counters
        self.train_volume_counts = defaultdict(int)
        self.val_volume_counts   = defaultdict(int)

        for idx in train_idx:
            (_pid, label) = self.all_volumes[idx]
            cls_name = self.classes[label]
            self.train_volume_counts[cls_name] += 1

        for idx in val_idx:
            (_pid, label) = self.all_volumes[idx]
            cls_name = self.classes[label]
            self.val_volume_counts[cls_name]   += 1

        train_sum = sum(self.train_volume_counts.values())
        val_sum   = sum(self.val_volume_counts.values())
        self.logger.info(f"Train volumes: {train_sum}, Val volumes: {val_sum}")

        if self.total_volumes > 0:
            for cls in self.classes:
                self.train_volume_percent[cls] = (self.train_volume_counts[cls]/self.total_volumes)*100
                self.val_volume_percent[cls]   = (self.val_volume_counts[cls]/self.total_volumes)*100

    def get_statistics(self):
        """
        Build 5 main tables:

        1) Slice-Based Overall (the 115,260 total slices)
        2) Slice-Based Train/Val
        3) Patient Counts
        4) Volume-Based Overall (the 3,842 total volumes)
        5) Volume-Based Train/Val (the 3,073 vs 769 approach)
        """

        # Table 1: Slice-Based Overall
        # e.g.
        # | CP   | CP %   | NCP  | NCP %   | Normal | Normal % | Total  | Total % |
        slice_headers1 = []
        slice_row1 = []
        for cls in self.classes:
            slice_headers1.extend([cls, f"{cls} %"])
            slice_row1.append(self.slice_class_counts[cls])
            slice_row1.append(f"{self.slice_class_percent.get(cls,0.0):.2f}%")
        slice_headers1.extend(["Total", "Total %"])
        slice_row1.append(self.total_slices)
        slice_row1.append("100.00%")
        slice_overall_table = tabulate([slice_row1], headers=slice_headers1, tablefmt="pipe")

        # Table 2: Slice-Based Train/Val
        # | Split | CP   | CP %  | NCP   | NCP %  | Normal | Normal % | Total | Total % |
        slice_headers2 = ["Split"]
        for cls in self.classes:
            slice_headers2.extend([cls, f"{cls} %"])
        slice_headers2.extend(["Total", "Total %"])

        # build train row
        train_row = ["Train"]
        for cls in self.classes:
            train_row.append(self.train_slice_counts[cls])
            train_row.append(f"{self.train_slice_percent.get(cls,0.0):.2f}%")
        train_sum = sum(self.train_slice_counts.values())
        train_row.append(train_sum)
        train_row.append(f"{(train_sum/self.total_slices*100):.2f}%" if self.total_slices>0 else "0%")

        # build val row
        val_row = ["Val"]
        for cls in self.classes:
            val_row.append(self.val_slice_counts[cls])
            val_row.append(f"{self.val_slice_percent.get(cls,0.0):.2f}%")
        val_sum = sum(self.val_slice_counts.values())
        val_row.append(val_sum)
        val_row.append(f"{(val_sum/self.total_slices*100):.2f}%" if self.total_slices>0 else "0%")

        slice_trainval_table = tabulate([train_row, val_row], headers=slice_headers2, tablefmt="pipe")

        # Table 3: Patient Counts
        # e.g.
        # | Class  | Total Patients | % Patients | Train Patients | Train % | Val Patients | Val % |
        patient_headers = ["Class", "Total Patients", "% Patients", "Train Patients", "Train %", "Val Patients", "Val %"]
        patient_rows = []
        for cls in self.classes:
            tot_p = self.patients_per_class[cls]
            if self.total_patients>0:
                tot_p_perc = f"{(tot_p/self.total_patients)*100:.2f}%"
            else:
                tot_p_perc="0.00%"
            tr_p = self.train_patients_count[cls]
            val_p= self.val_patients_count[cls]
            tr_perc = f"{(tr_p/self.total_patients)*100:.2f}%" if self.total_patients>0 else "0%"
            val_perc= f"{(val_p/self.total_patients)*100:.2f}%" if self.total_patients>0 else "0%"
            patient_rows.append([cls, tot_p, tot_p_perc, tr_p, tr_perc, val_p, val_perc])
        # total row
        patient_rows.append([
            "Total",
            self.total_patients,
            "100.00%",
            sum(self.train_patients_count.values()),
            f"{(sum(self.train_patients_count.values())/self.total_patients*100):.2f}%" if self.total_patients>0 else "0%",
            sum(self.val_patients_count.values()),
            f"{(sum(self.val_patients_count.values())/self.total_patients*100):.2f}%" if self.total_patients>0 else "0%"
        ])
        patient_table = tabulate(patient_rows, headers=patient_headers, tablefmt="pipe")

        # Table 4: Volume-Based Overall
        # e.g. 
        # | CP   | CP %  | NCP   | NCP %  | Normal | Normal % | Total | Total % |
        vol_headers1 = []
        vol_row1 = []
        for cls in self.classes:
            vol_headers1.extend([cls, f"{cls} %"])
            vol_count = self.volume_class_counts[cls]
            vol_percent= self.volume_class_percent.get(cls,0.0)
            vol_row1.append(vol_count)
            vol_row1.append(f"{vol_percent:.2f}%")
        vol_headers1.extend(["Total", "Total %"])
        vol_row1.append(self.total_volumes)
        vol_row1.append("100.00%")
        volume_overall_table = tabulate([vol_row1], headers=vol_headers1, tablefmt="pipe")

        # Table 5: Volume-Based Train/Val
        # like your final approach: 3,073 vs 769
        vol_headers2 = ["Class", "Train Volumes", "Train % of Total", "Val Volumes", "Val % of Total"]
        vol_rows2 = []
        for cls in self.classes:
            tr = self.train_volume_counts[cls]
            tr_perc = self.train_volume_percent.get(cls,0.0)
            va = self.val_volume_counts[cls]
            va_perc = self.val_volume_percent.get(cls,0.0)
            vol_rows2.append([cls, tr, f"{tr_perc:.2f}%", va, f"{va_perc:.2f}%"])
        # total row
        train_total = sum(self.train_volume_counts.values())
        val_total   = sum(self.val_volume_counts.values())
        train_total_perc= f"{(train_total/self.total_volumes*100):.2f}%" if self.total_volumes>0 else "0%"
        val_total_perc  = f"{(val_total/self.total_volumes*100):.2f}%"   if self.total_volumes>0 else "0%"
        vol_rows2.append(["Total", train_total, train_total_perc, val_total, val_total_perc])
        volume_trainval_table = tabulate(vol_rows2, headers=vol_headers2, tablefmt="pipe")

        return (
            slice_overall_table,     # Table 1
            slice_trainval_table,    # Table 2
            patient_table,           # Table 3
            volume_overall_table,    # Table 4
            volume_trainval_table    # Table 5
        )

def main():
    parser = argparse.ArgumentParser(description="Dataset Statistics combining slice-based & volume-based approaches.")
    parser.add_argument("dataset_root", type=str, help="Path to dataset root, e.g. ./data/ccccii")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for StratifiedKFold (default=5)")
    parser.add_argument("--i", type=int, default=0, help="Fold index (0-based), default=0")
    parser.add_argument("--volume_size", type=int, default=30, help="Min slices => 1 volume (default=30)")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        sys.exit(f"Error: {args.dataset_root} not found")

    stats = DatasetStatistics(dataset_root=args.dataset_root, k=args.k, i=args.i, volume_size=args.volume_size)
    (
        slice_overall_table,
        slice_trainval_table,
        patient_table,
        volume_overall_table,
        volume_trainval_table
    ) = stats.get_statistics()

    print("## Slice-Based Overall Statistics\n")
    print(slice_overall_table)
    print("\n## Slice-Based Train/Val Statistics\n")
    print(slice_trainval_table)
    print("\n## Patient Counts\n")
    print(patient_table)
    print("\n## Volume-Based Overall Statistics\n")
    print(volume_overall_table)
    print("\n## Volume-Based Train/Val Statistics\n")
    print(volume_trainval_table)

if __name__ == "__main__":
    main()


# usage:
# python -m datasets.dataset_statistics ./data/ccccii