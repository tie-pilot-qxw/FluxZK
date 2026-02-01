def extract_cost_time_simple(content):
    times = []
    for line in content.split('\n'):
        if 'Total cost time:' in line:
            # Split the string and take the last part
            time_str = line.split('Total cost time:')[1].strip()
            times.append(float(time_str))
    return times

if __name__ == "__main__":
    file_path = input("Enter the file path: ")
    with open(file_path, 'r') as file:
        content = file.read()
    times = extract_cost_time_simple(content)
    print("Extracted times:", times)
