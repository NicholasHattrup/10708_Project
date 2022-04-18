import os
for subdir, dirs, files in os.walk("xyz_inputs/"):
    for n,filename in enumerate(files):
            lines = []
            with open("xyz_inputs/"+filename, 'r') as f:
                lines = f.readlines()
#            filename_line = lines[1]
#            filename_line = filename_line[0] + '_' + filename_line[1:]
#            lines[1] = filename_line

            lines[-1], lines[-2] = lines[-2], lines[-1]
#            print(lines)
            with open("xyz_inputs_2/"+filename, "w") as f:
                for line in lines:
                    f.write(line)
                    if line == 'no frequencies':
                        f.write("\n")
            if n % 10000 == 0:
                print(n)
                    
