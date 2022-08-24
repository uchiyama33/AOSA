import shutil

if __name__ == "__main__":
    ###
    target_file = "pyflow/src/OpticalFlow.cpp"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if i == 11:
            lines[i] = "bool OpticalFlow::IsDisplay=false;\n"
        if i == 12:
            lines[i] = "\n"
        if i == 13:
            lines[i] = "\n"
        if i == 14:
            lines[i] = "\n"
        if i == 15:
            lines[i] = "\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)
