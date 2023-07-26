import os
import pyverilog.vparser.ast as vast

# Function to recursively traverse the abstract syntax tree and build the syntax vector
def build_syntax_vector(node, syntax_vector):
    syntax_vector.append(node.__class__.__name__)
    for child in node.children():
        if isinstance(child, vast.Node):
            build_syntax_vector(child, syntax_vector)
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, vast.Node):
                    build_syntax_vector(item, syntax_vector)

# Function to parse Verilog code and return the syntax vector
def parse_verilog_to_syntax_vector(verilog_code):
    ast = vast.parse(verilog_code)
    syntax_vector = []
    build_syntax_vector(ast, syntax_vector)
    return syntax_vector

if __name__ == "__main__":
    # Directory containing the .v files
    sample_dir = "AES-T_sample"

    # Loop through all .v files in the directory
    for i in range(600):
        filename = os.path.join(sample_dir, f"{i}.v")

        with open(filename, "r") as f:
            verilog_code = f.read()

        syntax_vector = parse_verilog_to_syntax_vector(verilog_code)

        # Save the syntax vector to a file with the same name as the .v file
        with open(f"{i}_syntax_vector.txt", "w") as f:
            for item in syntax_vector:
                f.write(item + "\n")

        print(f"Syntax vector for {filename} saved to {i}_syntax_vector.txt")