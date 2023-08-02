import os
from pyverilog.vparser.parser import parse
import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")  # Prevent encoding errors

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def save_ast_to_file(ast_node, file_path):
    # Save the AST to a file
    with open(file_path, 'w', encoding="utf-8") as f:
        ast_node.show(buf=f)

def main():
    input_dir = '../data/AES-T_Sample'
    output_dir = '../data/AES-T_Sequence'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Redirect the stdout to the file "ast_output.txt"
    log_file = os.path.join(output_dir, "ast_output.txt")
    sys.stdout = Logger(log_file)

    for idx, file_name in enumerate(os.listdir(input_dir)):
        if file_name.endswith('.v'):
            file_path = os.path.join(input_dir, file_name)
            ast, _ = parse([file_path])

            # Display AST on the console
            ast.show()

            # Save AST to a separate output file
            output_file = os.path.splitext(file_name)[0] + "_ast_output.txt"
            output_file_path = os.path.join(output_dir, output_file)
            save_ast_to_file(ast, output_file_path)

    # Reset stdout back to the console
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    main()
