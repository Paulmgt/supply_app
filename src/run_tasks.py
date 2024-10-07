import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Attend que le processus se termine
    return process.returncode

def main():
    commands = [
        "python /src/clean/scrapper.py",
        "python /src/clean/clean_a.py",
        "python /src/clean/clean_b.py",
        "python /src/utils/train.py"
    ]

    for command in commands:
        return_code = run_command(command)
        if return_code != 0:
            print(f"Erreur lors de l'exécution de {command}")
            break  # Stoppe si une commande échoue

if __name__ == "__main__":
    main()
