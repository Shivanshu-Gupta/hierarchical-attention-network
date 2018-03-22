def log(output, outfile=None):
    print(output)
    if outfile is not None:
        with open(outfile, 'a') as outf:
            outf.write(str(output) + '\n')
            outf.flush()

def result_json_to_csv(exps):
    import csv
    import json
    from os.path import join, exists
    combined_results = []
    for exp in exps:
        combined_results.append([exp])
        jsonfile = join(exp, 'all_scalars.json')
        csvfile = join(exp, 'results.csv')
        if exists(jsonfile):
            print('{} -> {}'.format(jsonfile, csvfile))
            res = json.load(open(jsonfile))
            iters = len(res['Train Loss'])
            f = open(csvfile, 'w')
            writer = csv.writer(f)
            results = []
            results.append(list(res.keys()))
            for i in range(iters):
                results.append([res[key][i][2] for key in res.keys()])
            combined_results += results
            writer.writerows(results)
    f = open('combined_results.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(combined_results)
