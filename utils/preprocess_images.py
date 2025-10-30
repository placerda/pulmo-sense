"""
Script para pré-processar imagens PNG:
- Rotaciona 90 graus no sentido anti-horário
- Mantém apenas as 30 imagens centrais de cada diretório
"""

import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from log_config import get_custom_logger

logger = get_custom_logger(__name__)


def rotate_image(image_path, angle=-90, output_path=None):
    """
    Rotaciona uma imagem pelo ângulo especificado.
    
    Args:
        image_path: Caminho da imagem de entrada
        angle: Ângulo de rotação (negativo = anti-horário)
        output_path: Caminho de saída (se None, sobrescreve a original)
    
    Returns:
        True se sucesso, False caso contrário
    """
    try:
        img = Image.open(image_path)
        rotated_img = img.rotate(angle, expand=True)
        
        if output_path is None:
            output_path = image_path
        
        rotated_img.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Erro ao rotacionar {image_path}: {str(e)}")
        return False


def get_central_images(image_files, num_images=30):
    """
    Retorna os índices das imagens centrais.
    
    Args:
        image_files: Lista de arquivos de imagem
        num_images: Número de imagens centrais a manter
    
    Returns:
        Lista de arquivos das imagens centrais
    """
    total = len(image_files)
    
    if total <= num_images:
        return image_files
    
    # Calcula o índice inicial e final para pegar as imagens centrais
    start_idx = (total - num_images) // 2
    end_idx = start_idx + num_images
    
    return image_files[start_idx:end_idx]


def process_directory(base_dir, num_central_images=30, rotate_angle=-90, 
                     output_dir=None, backup=False):
    """
    Processa todos os diretórios de imagens.
    
    Args:
        base_dir: Diretório base contendo as pastas de imagens
        num_central_images: Número de imagens centrais a manter (padrão: 30)
        rotate_angle: Ângulo de rotação (padrão: -90 = anti-horário)
        output_dir: Diretório de saída (se None, modifica in-place)
        backup: Se True, cria backup antes de processar
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        logger.error(f"Diretório não encontrado: {base_dir}")
        return
    
    # Encontra todos os diretórios que contêm imagens
    # Padrão: NCP/<nome_caso>/001/
    case_dirs = []
    
    for category_dir in base_path.iterdir():
        if category_dir.is_dir():
            for case_dir in category_dir.iterdir():
                if case_dir.is_dir():
                    for slice_dir in case_dir.iterdir():
                        if slice_dir.is_dir():
                            case_dirs.append(slice_dir)
    
    if not case_dirs:
        logger.warning(f"Nenhum diretório de imagens encontrado em {base_dir}")
        return
    
    logger.info(f"Encontrados {len(case_dirs)} diretórios para processar")
    logger.info(f"Configuração: Rotação={rotate_angle}°, Imagens centrais={num_central_images}")
    
    total_images_before = 0
    total_images_after = 0
    successful_dirs = 0
    failed_dirs = 0
    
    # Processa cada diretório
    for slice_dir in tqdm(case_dirs, desc="Processando diretórios"):
        try:
            # Lista todas as imagens PNG no diretório
            image_files = sorted(list(slice_dir.glob("*.png")))
            
            if not image_files:
                logger.warning(f"Nenhuma imagem PNG encontrada em {slice_dir}")
                continue
            
            total_images_before += len(image_files)
            
            # Seleciona as imagens centrais
            central_images = get_central_images(image_files, num_central_images)
            
            # Cria diretório de saída se especificado
            if output_dir:
                output_path = Path(output_dir) / slice_dir.relative_to(base_path)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = slice_dir
            
            # Cria backup se solicitado
            if backup and not output_dir:
                backup_dir = slice_dir.parent / f"{slice_dir.name}_backup"
                if not backup_dir.exists():
                    shutil.copytree(slice_dir, backup_dir)
                    logger.debug(f"Backup criado: {backup_dir}")
            
            # Processa as imagens centrais
            processed_images = []
            for img_file in central_images:
                if output_dir:
                    output_file = output_path / img_file.name
                else:
                    output_file = img_file
                
                if rotate_image(img_file, angle=rotate_angle, output_path=output_file):
                    processed_images.append(output_file)
            
            # Se não está usando output_dir, remove as imagens que não são centrais
            if not output_dir:
                images_to_remove = set(image_files) - set(central_images)
                for img_file in images_to_remove:
                    img_file.unlink()
            
            total_images_after += len(processed_images)
            successful_dirs += 1
            
            logger.info(
                f"✓ {slice_dir.parent.name}/{slice_dir.name}: "
                f"{len(image_files)} → {len(processed_images)} imagens"
            )
            
        except Exception as e:
            logger.error(f"✗ Erro ao processar {slice_dir}: {str(e)}")
            failed_dirs += 1
    
    # Resumo final
    logger.info(f"\n{'='*60}")
    logger.info(f"Processamento concluído!")
    logger.info(f"Diretórios processados com sucesso: {successful_dirs}")
    logger.info(f"Diretórios com falha: {failed_dirs}")
    logger.info(f"Total de imagens antes: {total_images_before}")
    logger.info(f"Total de imagens depois: {total_images_after}")
    logger.info(f"Imagens removidas: {total_images_before - total_images_after}")
    logger.info(f"Redução: {(1 - total_images_after/total_images_before)*100:.1f}%")
    logger.info(f"{'='*60}")


def main():
    """Função principal com argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Pré-processa imagens PNG: rotaciona e seleciona imagens centrais'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../data/covid19ctpng',
        help='Diretório de entrada contendo as imagens (padrão: ../data/covid19ctpng)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Diretório de saída (se não especificado, modifica in-place)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=30,
        help='Número de imagens centrais a manter (padrão: 30)'
    )
    parser.add_argument(
        '--rotation',
        type=int,
        default=-90,
        help='Ângulo de rotação em graus, negativo=anti-horário (padrão: -90)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Cria backup antes de processar (apenas para modificação in-place)'
    )
    parser.add_argument(
        '--no-rotation',
        action='store_true',
        help='Desabilita a rotação, apenas seleciona imagens centrais'
    )
    
    args = parser.parse_args()
    
    rotation_angle = 0 if args.no_rotation else args.rotation
    
    logger.info("="*60)
    logger.info("Iniciando pré-processamento de imagens")
    logger.info("="*60)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir if args.output_dir else 'in-place'}")
    logger.info(f"Imagens centrais: {args.num_images}")
    logger.info(f"Rotação: {rotation_angle}°")
    logger.info(f"Backup: {args.backup}")
    logger.info("="*60)
    
    # Confirmação para operação in-place
    if not args.output_dir:
        logger.warning("\n⚠️  ATENÇÃO: Operação in-place irá MODIFICAR os arquivos originais!")
        if not args.backup:
            logger.warning("⚠️  Backup não está habilitado. Use --backup para criar backup.")
        
        response = input("\nDeseja continuar? (sim/não): ")
        if response.lower() not in ['sim', 's', 'yes', 'y']:
            logger.info("Operação cancelada pelo usuário.")
            return
    
    process_directory(
        base_dir=args.input_dir,
        num_central_images=args.num_images,
        rotate_angle=rotation_angle,
        output_dir=args.output_dir,
        backup=args.backup
    )


if __name__ == '__main__':
    main()
