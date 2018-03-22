def log(output, outfile=None):
    print(output)
    if outfile is not None:
        with open(outfile, 'a') as outf:
            outf.write(str(output) + '\n')
            outf.flush()