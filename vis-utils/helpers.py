import numpy as np

def if_match(source, own_keys, dict_):
    found = 0
    
    for tkey_1 in dict_.keys():
        for tkey_2 in dict_[tkey_1].keys():
            for idx, target in enumerate(dict_[tkey_1][tkey_2]):
                if(tkey_1 == own_keys[0] and tkey_2 == own_keys[1]):
                    continue
                
                if(np.array_equal(source, target)):
                    print(f"conflict in {own_keys[0]},{own_keys[1]},{own_keys[2]} == {tkey_1},{tkey_2},{idx}")
                    found += 1
    

    if(found>0):
        print(f"{found} conflicts")
        return 1           
    
def remove_duplicates(dataset):

    new_unique_waves = {}

    for skey_1 in dataset.keys():
        for skey_2 in dataset[skey_1].keys():
            for idx, source in enumerate(dataset[skey_1][skey_2]):

                if(if_match(source,[skey_1, skey_2, idx], new_unique_waves)):
                    continue
                else:
                    if(skey_1 in new_unique_waves.keys()):

                        if(skey_2 in new_unique_waves[skey_1].keys()):
                            new_unique_waves[skey_1][skey_2].append(source)
                        else:
                            new_unique_waves[skey_1][skey_2] = []
                            new_unique_waves[skey_1][skey_2].append(source)

                    else:
                        new_unique_waves[skey_1] = {}
                        new_unique_waves[skey_1][skey_2] = []
                        new_unique_waves[skey_1][skey_2].append(source)
                        
    return new_unique_waves