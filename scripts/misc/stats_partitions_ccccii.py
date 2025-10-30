#!/usr/bin/env python3
"""
Conta pacientes e volumes por particao do CC-CCII selecionado.

Percorre as particoes abaixo (se existirem dentro de --data-root):
  - ccccii_selected_train
  - ccccii_selected_test
  - ccccii_selected_fold_{N}_train
  - ccccii_selected_fold_{N}_val

Para cada particao imprime:
  Common Pneumonia - a pacientes, b Volumes
  Novel COVID Pneumonia - c pacientes, d Volumes
  Normal - e pacientes, f Volumes
  Total - g pacientes, h Volumes
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

# Ordem de exibicao das classes (se nao existirem, sao ignoradas)
CLASS_ORDER = ["Common Pneumonia", "Novel COVID Pneumonia", "Normal"]

def contar_pacientes_e_volumes(split_dir: Path) -> Dict[str, Tuple[int, int]]:
    """
    Retorna dict {classe: (num_pacientes, num_volumes)} para a particao dada.
    - num_pacientes: subpastas imediatas em split_dir/classe
    - num_volumes: soma das subpastas imediatas dentro de cada paciente
    """
    resultado: Dict[str, Tuple[int, int]] = {}
    if not split_dir.exists() or not split_dir.is_dir():
        return resultado

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        classe = class_dir.name
        # pacientes = subpastas imediatas
        pacientes = [p for p in class_dir.iterdir() if p.is_dir()]
        num_pacientes = len(pacientes)

        # volumes = soma das subpastas imediatas dentro de cada paciente
        num_volumes = 0
        for p in pacientes:
            scans = [s for s in p.iterdir() if s.is_dir()]
            num_volumes += len(scans)

        resultado[classe] = (num_pacientes, num_volumes)

    return resultado


def split_sort_key(name: str):
    """
    Ordena para imprimir primeiro base train/test, depois folds em ordem:
      ccccii_selected_train
      ccccii_selected_test
      ccccii_selected_fold_1_train
      ccccii_selected_fold_1_val
      ccccii_selected_fold_2_train
      ...
    """
    if name == "ccccii_selected_train":
        return (0, 0, 0)
    if name == "ccccii_selected_test":
        return (0, 0, 1)
    m = re.match(r"^ccccii_selected_fold_(\d+)_(train|val)$", name)
    if m:
        fold = int(m.group(1))
        phase = 0 if m.group(2) == "train" else 1
        return (1, fold, phase)
    # outros/nao padrao
    return (2, name, 0)


def main():
    parser = argparse.ArgumentParser(description="Totaliza pacientes e volumes por particao do CC-CCII selecionado")
    parser.add_argument(
        "-d", "--data-root",
        default=r"\\wsl.localhost\Ubuntu\home\paulo\workspace\doutorado\data",
        help="Diretorio 'data' onde ficam as particoes (padrao: C:\\Users\\paulolacerda\\workspace\\doutorado\\data)"
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERRO] Diretorio nao existe: {data_root}")
        return

    # Lista candidatos de particao que comecam com 'ccccii_selected'
    splits = [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("ccccii_selected")]
    splits_sorted = sorted(splits, key=lambda p: split_sort_key(p.name))

    if not splits_sorted:
        print(f"Nenhuma particao encontrada em: {data_root}")
        return

    for split_path in splits_sorted:
        nome = split_path.name
        stats = contar_pacientes_e_volumes(split_path)

        # Prepara linhas no formato pedido
        print(f"\n=== {nome} ===")

        total_pac = 0
        total_vol = 0

        # Mostra nas classes em ordem definida; se uma classe nao existir, pula
        for classe in CLASS_ORDER:
            if classe in stats:
                pac, vol = stats[classe]
                total_pac += pac
                total_vol += vol
                print(f"{classe} - {pac} pacientes, {vol} Volumes")

        # Tambem imprime classes que nao estao em CLASS_ORDER (se houver)
        outras = [c for c in stats.keys() if c not in CLASS_ORDER]
        for classe in sorted(outras):
            pac, vol = stats[classe]
            total_pac += pac
            total_vol += vol
            print(f"{classe} - {pac} pacientes, {vol} Volumes")

        print(f"Total - {total_pac} pacientes, {total_vol} Volumes")


if __name__ == "__main__":
    main()
