import os

vault_path = "C:/Users/bruno/Desktop/D&D/Obsidian/Aquarius"
output_file = "compiled_setting.md"

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, vault_path)

                outfile.write(f"# {rel_path}\n\n")  # section header = filename
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n---\n\n")
