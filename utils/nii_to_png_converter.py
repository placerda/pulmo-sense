"""
Script para converter arquivos NIfTI (.nii) para imagens PNG.
Estrutura de saída: ../data/covid19ctpng/NCP/<nome_arquivo>/<slice_number>/<slice_xxx>.png
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from log_config import get_custom_logger

logger = get_custom_logger(__name__)


def normalize_image(image_array):
    """
    Normaliza a imagem para o intervalo [0, 255].
    
    Args:
        image_array: Array numpy com os dados da imagem
        
    Returns:
        Array numpy normalizado
    """
    # Remove valores extremos (outliers)
    min_val = np.percentile(image_array, 1)
    max_val = np.percentile(image_array, 99)
    
    # Clip valores
    image_array = np.clip(image_array, min_val, max_val)
    
    # Normaliza para [0, 255]
    if max_val - min_val > 0:
        image_array = ((image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        image_array = np.zeros_like(image_array, dtype=np.uint8)
    
    return image_array


def convert_nii_to_png(nii_file_path, output_base_dir, category='NCP'):
    """
    Converte um arquivo NIfTI para múltiplas imagens PNG (uma por corte/slice).
    
    Args:
        nii_file_path: Caminho completo para o arquivo .nii
        output_base_dir: Diretório base de saída
        category: Categoria da imagem (padrão: 'NCP')
    """
    try:
        # Carrega o arquivo NIfTI
        nii_img = nib.load(nii_file_path)
        nii_data = nii_img.get_fdata()
        
        # Obtém o nome do arquivo sem extensão
        file_name = Path(nii_file_path).stem
        if file_name.endswith('.nii'):
            file_name = file_name[:-4]  # Remove .nii se presente
        
        # Cria o diretório de saída
        output_dir = Path(output_base_dir) / category / file_name / '001'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determina o eixo dos cortes (geralmente o último eixo)
        num_slices = nii_data.shape[2]
        
        logger.info(f"Processando {file_name}: {num_slices} cortes")
        
        # Itera sobre cada corte
        for slice_idx in range(num_slices):
            # Extrai o corte
            slice_data = nii_data[:, :, slice_idx]
            
            # Normaliza a imagem
            slice_normalized = normalize_image(slice_data)
            
            # Converte para imagem PIL
            img = Image.fromarray(slice_normalized, mode='L')
            
            # Salva como PNG
            output_file = output_dir / f"slice_{slice_idx:03d}.png"
            img.save(output_file)
        
        logger.info(f"✓ {file_name}: {num_slices} imagens PNG geradas em {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro ao processar {nii_file_path}: {str(e)}")
        return False


def process_directory(input_dir, output_dir, category='NCP', file_pattern='*.nii'):
    """
    Processa todos os arquivos .nii em um diretório.
    
    Args:
        input_dir: Diretório de entrada contendo arquivos .nii
        output_dir: Diretório base de saída
        category: Categoria da imagem (padrão: 'NCP')
        file_pattern: Padrão de arquivo para buscar (padrão: '*.nii')
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Diretório de entrada não encontrado: {input_dir}")
        return
    
    # Lista todos os arquivos .nii
    nii_files = list(input_path.glob(file_pattern))
    
    if not nii_files:
        logger.warning(f"Nenhum arquivo {file_pattern} encontrado em {input_dir}")
        return
    
    logger.info(f"Encontrados {len(nii_files)} arquivos para processar")
    
    # Processa cada arquivo
    successful = 0
    failed = 0
    
    for nii_file in tqdm(nii_files, desc="Convertendo arquivos"):
        if convert_nii_to_png(nii_file, output_dir, category):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Processamento concluído!")
    logger.info(f"Sucesso: {successful} arquivos")
    logger.info(f"Falhas: {failed} arquivos")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*50}")


def main():
    """Função principal com argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Converte arquivos NIfTI (.nii) para imagens PNG'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../data/covid19ct',
        help='Diretório de entrada contendo arquivos .nii (padrão: ../data/covid19ct)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/covid19ctpng',
        help='Diretório de saída para imagens PNG (padrão: ../data/covid19ctpng)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='NCP',
        help='Categoria da imagem (padrão: NCP)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.nii',
        help='Padrão de arquivo para buscar (padrão: *.nii)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Iniciando conversão NIfTI para PNG")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Category: {args.category}")
    logger.info(f"Pattern: {args.pattern}")
    
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        category=args.category,
        file_pattern=args.pattern
    )


if __name__ == '__main__':
    main()