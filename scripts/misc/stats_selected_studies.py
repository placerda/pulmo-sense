"""
Resumo de estudos selecionados

Este script lê um arquivo de caminhos (um por linha) e gera um resumo
com o número de pacientes únicos e número total de scans por categoria.

Uso:
    python summarize_selected_studies.py [--file PATH]

Exemplo:
    python summarize_selected_studies.py --file "local\organize\data_analysis\selected-nonsegmented.txt"

(Ou para usar o arquivo original:)
    python summarize_selected_studies.py --file "local\organize\data_analysis\selected-studies.txt"

O script assume que cada linha tem um padrão de caminho com pelo menos
3 partes finais: .../<categoria>/<paciente>/<scan>
Ex.: data/ccccii/Normal/1728/1016

Saída:
    Categoria: pacientes=<n> scans=<m>

"""

from collections import defaultdict
import argparse
import os
import sys


def summarize(file_path):
    patients_by_cat = defaultdict(set)
    scans_by_cat = defaultdict(int)

    if not os.path.isfile(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return 1

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Normaliza separadores e divide em partes
            parts = os.path.normpath(line).split(os.sep)

            if len(parts) < 3:
                print(f"Linha {i}: formato inesperado (menos de 3 partes): '{line}'", file=sys.stderr)
                continue

            # Assume as três últimas partes: categoria/paciente/scan
            category, patient, scan = parts[-3], parts[-2], parts[-1]

            patients_by_cat[category].add(patient)
            scans_by_cat[category] += 1

    # Imprime resumo ordenado por categoria
    categories = sorted(set(list(patients_by_cat.keys()) + list(scans_by_cat.keys())))

    total_patients = set()
    total_scans = 0

    print("Resumo por categoria:\n")
    for cat in categories:
        n_patients = len(patients_by_cat.get(cat, set()))
        n_scans = scans_by_cat.get(cat, 0)
        total_patients.update(patients_by_cat.get(cat, set()))
        total_scans += n_scans
        print(f"{cat}: pacientes={n_patients} scans={n_scans}")

    print("\nTotais:")
    print(f"categorias={len(categories)} pacientes_unicos={len(total_patients)} scans_totais={total_scans}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume patients and scans per category from a list of paths.")
    parser.add_argument("--file", "-f", default=r"local\organize\data_analysis\selected-segmented.txt",
                        help="Caminho para o arquivo de estudos (um caminho por linha). Pode passar 'local\\organize\\data_analysis\\selected-studies.txt' se quiser analisar o arquivo original.")
    args = parser.parse_args()

    sys.exit(summarize(args.file))
