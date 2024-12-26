import os
import ast
import subprocess
import sys
import requests

def get_package_info(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        package_info = response.json()
        return package_info
    else:
        return None

def extract_imports(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    imports = {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)}
    imports.update({node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)})
    return imports

def get_installed_packages():
    installed = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    return {pkg.split('==')[0] for pkg in installed.decode().split()}

def read_requirements(requirements_path):
    with open(requirements_path, "r") as file:
        return {line.split("==")[0].strip() for line in file.readlines()}

def map_import_to_package(import_name):
    package_info = get_package_info(import_name)
    if package_info:
        return package_info["info"]["name"]
    return import_name

def main(requirements_path, source_folder):
    requirements = read_requirements(requirements_path)
    installed_packages = get_installed_packages()
    
    python_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_folder) for f in filenames if f.endswith(".py")]
    all_imports = set()
    
    for python_file in python_files:
        all_imports.update(extract_imports(python_file))
    
    missing_packages = {map_import_to_package(imp) for imp in all_imports - requirements - installed_packages - set(sys.builtin_module_names)}
    
    if missing_packages:
        print("Missing packages:")
        for package in missing_packages:
            print(package)
        
        print("\nRun the following command to install the missing packages:")
        print("pip install " + " ".join(missing_packages))
        
        print("\nAfter installation, you can check the versions of these packages using:")
        print("pip show " + " ".join(missing_packages))
    else:
        print("All necessary packages are listed in requirements.txt or are already installed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_requirements.py <path_to_requirements.txt> <source_folder>")
        sys.exit(1)
    
    requirements_path = sys.argv[1]
    source_folder = sys.argv[2]
    
    main(requirements_path, source_folder)
