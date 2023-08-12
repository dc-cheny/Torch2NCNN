import os

import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class FoodEmbeddingEval:
    def __init__(self, library_dir, mode='cluster', dist='l2'):
        self.mode = mode
        self.library_dir = library_dir
        self.dist = dist

    def cluster_eval(self):
        embeddings, class_indexes = self.read_vectors_from_npz()
        emb_dict = {}

        for idx, ci in enumerate(class_indexes):
            if ci not in emb_dict:
                emb_dict[ci] = [embeddings[idx]]
            else:
                emb_dict[ci].append(embeddings[idx])

        res = {c: [] for c in class_indexes}

        mutual_sim = cosine_similarity(embeddings, embeddings)

        for _idx, pc in tqdm(enumerate(class_indexes)):
            dist_name_list = []
            inter_class_embed = emb_dict[pc]
            if len(inter_class_embed) == 1:
                continue

            for i in range(len(mutual_sim)):
                if i == _idx:
                    continue
                dist_name_list.append((mutual_sim[_idx][i], class_indexes[i]))
            dist_name_list.sort(reverse=True if self.dist == 'cosine' else False)
            dist_name_list = dist_name_list[:len(inter_class_embed) - 1]
            res[pc].append(sum([x[1] == pc for x in dist_name_list]) / (len(inter_class_embed) - 1))

        final_score_list = []
        for _rk, _rv in res.items():
            if not _rv:
                continue
            _r_scores = sum(_rv) / len(_rv)
            # print('The score of {} is: {}'.format(_rk, _r_scores))
            final_score_list.append(_r_scores)

        final_score = sum(final_score_list) / len(final_score_list)
        print('Total score is: {}.'.format(final_score))
        return final_score

    def calc_dist(self, v1, v2):
        if self.dist == 'l2':
            return np.linalg.norm(v1 - v2)
        elif self.dist == 'cosine':
            return distance.cosine(v1, v2)
        return None

    def read_vectors_from_npz(self):
        """ read vectors from txt file
        """
        embeddings = np.load(os.path.join(self.library_dir, 'embeddings.npy'))
        class_indexes = np.load(os.path.join(self.library_dir, 'class_indexes.npy'))
        return embeddings, class_indexes


def main():
    library_dir = r'C:\worksp\xxcy\Torch2NCNN\data\SeekerLibrary\eval_230806\mobilenetv3_large_20'

    dist = 'cosine'
    fee = FoodEmbeddingEval(library_dir=library_dir, dist=dist)
    fee.cluster_eval()


if __name__ == '__main__':
    main()
