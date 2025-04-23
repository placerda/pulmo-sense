#!/usr/bin/env python3

import argparse
import numpy as np

def main(file_path):
    # Carrega o arquivo .npy
    try:
        data = np.load(file_path)
        print(f"Tipo de dado (dtype dos voxels): {data.dtype}")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    print(f"Shape (dimensões) da matriz: {data.shape}")

    if data.ndim != 3:
        print("A matriz não é tridimensional.")
        return

    # Define os índices centrais
    depth = data.shape[2]
    if depth < 30:
        print(f"A matriz tem apenas {depth} fatias. Retornando todas.")
        central_slices = data
    else:
        start = (depth - 30) // 2
        end = start + 30
        central_slices = data[:, :, start:end]
        print(f"Selecionadas fatias centrais do índice {start} ao {end - 1}")

    print(f"Shape das 30 fatias centrais: {central_slices.shape}")

    # Estatísticas
    flattened = central_slices.flatten()
    print("\n📊 Estatísticas das 30 fatias centrais:")
    print(f"- Média: {flattened.mean():.2f}")
    print(f"- Desvio padrão: {flattened.std():.2f}")
    print(f"- Valor mínimo: {flattened.min()}")
    print(f"- Valor máximo: {flattened.max()}")

    count_minus_2048 = np.sum(flattened == -2048)
    total_voxels = flattened.size
    percentage_minus_2048 = 100 * count_minus_2048 / total_voxels
    print(f"- Voxels com valor -2048: {count_minus_2048} ({percentage_minus_2048:.2f}%)")

    # Mostra os primeiros 100 valores
    print("\nPrimeiros 100 valores (das fatias centrais):")
    print(flattened[:100])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspeciona um arquivo .npy 3D e mostra fatias centrais")
    parser.add_argument("file", help="Caminho para o arquivo .npy")
    args = parser.parse_args()
    main(args.file)
