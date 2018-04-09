import os

def log(output, outfile=None):
    print(output)
    if outfile is not None:
        with open(outfile, 'a') as outf:
            outf.write(str(output) + '\n')
            outf.flush()


options = {
    'optim': ['adam', 'sgd'],
    'vocab': ['comb', 'diff'],
    'size': ['large', 'small'],
    'rnn_type': ['lstm', 'gru'],
    'mlp': ['mlp', 'nomlp']
}


# Experiment Generation
def generate_exps():
    import yaml
    config = yaml.load(open('config_sample.yaml'))
    i = 0
    for optim in options['optim']:
        config['optim']['class'] = optim
        if optim == 'adam':
            config['optim']['params']['lr'] = 0.001
        else:
            config['optim']['params']['lr'] = 0.01
        for vocab in options['vocab']:
            if vocab == 'comb':
                config['data']['review_vocab'] = 'comb_vocab_pruned.pkl'
                config['data']['summary_vocab'] = 'comb_vocab_pruned.pkl'
            else:
                config['data']['review_vocab'] = 'review_vocab.pkl'
                config['data']['summary_vocab'] = 'summary_vocab.pkl'
            for size in options['size']:
                if size == 'large':
                    config['model']['params']['emb_dim'] = 200
                    config['model']['params']['rnn_hidden_dim'] = 100
                else:
                    config['model']['params']['emb_dim'] = 100
                    config['model']['params']['rnn_hidden_dim'] = 50
                for rnn in options['rnn_type']:
                    config['model']['params']['rnn_type'] = rnn
                    for mlp in options['mlp']:
                        if mlp == 'mlp':
                            config['model']['params']['use_summ_mlp'] = True
                        else:
                            config['model']['params']['use_summ_mlp'] = False
                        name = str(i) + '_config_' + '_'.join([optim, vocab, size, rnn, mlp])
                        yaml.dump(config, open(name, 'w'))
                        i += 1


# Experiment Analysis scripts
# example usage: result_json_to_csv(sorted(os.listdirs()))
def result_json_to_csv(exps):
    import csv
    import json
    from os.path import join, exists, isdir
    combined_results = []
    final_results = []
    for exp in exps:
        if not isdir(exp) or exp == 'temp':
            continue
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
            # results.append(list(['Train Fscore', 'Val Fscore']))
            for i in range(iters):
                # results.append([res[key][i][2] for key in res.keys()])
                results.append([res[key][i][2] for key in ['Train Fscore', 'Validation Fscore']])
            combined_results += results
            key = 'Validation Fscore'
            val_fscores = [res[key][i][2] for i in range(len(res[key]))]
            conf = [option[0] if option[0] in exp else option[1] for option in options]
            final_results.append(conf(exp) + [max(val_fscores)] + val_fscores)
            writer.writerows(results)
    writer = csv.writer(open('combined_results.csv', 'w'))
    writer.writerows(combined_results)
    writer = csv.writer(open('final_results.csv', 'w'))
    writer.writerows(final_results)
