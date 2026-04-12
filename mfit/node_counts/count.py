import sys

def count_nodes(file_path):
    nodes = set()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # skip comments or empty lines
            if not line or line.startswith("#"):
                continue

            u, v = map(int, line.split())
            nodes.add(u)
            nodes.add(v)

    return len(nodes)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_nodes.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    num_nodes = count_nodes(file_path)

    print(f"Number of nodes: {num_nodes}")
