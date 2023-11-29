
def topk_search_queryEmb(text_queryEmb, image_queryEmb, textIndex, imageIndex, k=20, multi_modal_mode = False, single_mode = False):
    if not multi_modal_mode:
        distances, indices = distance_search(text_queryEmb, textIndex, k, single_mode)

    else:
        t2t_distances, t2t_indices = distance_search(text_queryEmb, textIndex, k=int(k/4), single_mode=False)
        t2i_distances, t2i_indices = distance_search(text_queryEmb, imageIndex, k=int(k/4), single_mode=False)
        i2t_distances, i2t_indices = distance_search(image_queryEmb, textIndex, k=int(k/4), single_mode=False)
        i2i_distances, i2i_indices = distance_search(image_queryEmb, imageIndex, k=int(k/4), single_mode=False)
        
        distances, indices = multi_distance_search_result_gatherting(
            distances = [t2t_distances, t2i_distances, i2t_distances, i2i_distances],
            indices = [t2t_indices, t2i_indices, i2t_indices, i2i_indices],
            single_mode=single_mode
        )
    return distances, indices
        

def distance_search(query_emb, index, k=20, single_mode = False):
    """inner product search for query & embeddings

    Args:
        query_emb (Tensor): 유저의 쿼리에 대한 임베딩. 유저가 시스템에 입력한 자연어를 embedding으로 변형한 형태. Shape = (num_query, model_dim)
            주로 num_query = 1일 듯. 이 경우에는 shape=(model_dim)으로 입력해도 됨.
        index (faiss.swigfaiss.IndexFlatL2): 데이터베이스에 대한 FAISS임베딩. 
            번외: index 만드는 법.
            1. embedding = model.encode(sentences) 등등으로 shape:(num_data, model_dimension)을 갖는 float 텐서를 생성
            2. index = faiss.IndexFlatL2(embeddings.shape[1])
            3. index.add(embeddings)
            이렇게 만든 index를 넣어주면 됨.

        k (int, optional): 몇 개의 유사한 데이터를 뽑을 것인가. Defaults to 20.
        single_mode (bool, defaults to False): 
            faiss 함수들이 다들 배치연산(?)처럼 행동함.
            배치 고려 안 하고 단일 example에 대한 정보를 input/output으로 맞추고 싶다면 True로 전환하여 사용.
            single_mode == True이면,
                return값들인 distances와 indices는 list of list가 아니라 list로 반환됨.
        
    Returns:
        distances (list of list of float. ex: [[0.1, 0.2], [0.3, 0.4]]).
            distance[i]: i번째 쿼리에 대한 거리값을 담은 length = k의 리스트. 가까운 순으로 나열됨.
        indices (list of list of int. ex: [[1, 2], [3, 4]])
            indices[i]: i번째 쿼리에 대한 거리가 가까운 순으로 나열된 벡터들의 index를 표현함..
            
        distance example:
        [[0.         0.04974563 0.05171977 0.05315638 0.05881927 0.06011381 0.0609491 ]
         [0.         0.05888148 0.06278298 0.06477981 0.06501415 0.06592916 0.06735526]]
        
        index example:
        [[   0  511  692  123 1379 1726  192]
        [   1 1125 1786  199 1438 1096  103]]
        
    """
    
    if len(query_emb.shape) ==1:
        query_emb = query_emb.reshape(1, -1)
    if 'numpy' not in str(type(query_emb)):
        query_emb = query_emb.detach().numpy()
    # else:
    #     if not single_mode:
            # raise f"please give consistent information.\nCurrently, query_emb.shape={query_emb.shape} and single_mode={True}"
    print("query_emb:")
    print(f"type: {type(query_emb)}")
    print(f"shape: {query_emb.shape}")
    print(f"info: {query_emb}")
    distances, indices = index.search(query_emb.reshape(1, -1), k)
    distances = [list(d) for d in distances]
    indices = [list(d) for d in indices]
    if single_mode:
        distances = distances[0]
        indices = indices[0]
    return distances, indices


def multi_distance_search_result_gatherting(distances, indices, single_mode = False):
    # print("distances")
    # print(distances)
    # print("indices")
    # print(indices)
    # print("--------------")
    
    final_distance, final_indices = [[] for _ in range(len(distances))], [[] for _ in range(len(indices))]
    
    for f, distance_source in enumerate(distances):
        for e, dist in enumerate(distance_source):
            final_distance[f].append(dist)

    for f, index_source in enumerate(indices):
        for e, idx in enumerate(index_source):
            final_indices[f].append(idx)
    # print("final_indices")
    # print(final_indices)
    # print("final_distance")
    # print(final_distance)
            
    gathered = [[] for _ in range(len(distances))]
    for queryidx, g in enumerate(gathered):
        for dataidx, data in enumerate(final_distance[queryidx]):
            gathered[queryidx].append((final_distance[queryidx][dataidx], final_indices[queryidx][dataidx]))
        # print(f"gathered[queryidx]={gathered[queryidx]}")
        # gathered[queryidx].sort(key = lambda x:x[0])

    # print(f"len(gathered):{len(gathered)}")
    # print(f"len(gathered[0]):{len(gathered[0])}")
    # print(f"gathered[0]: {gathered[0]}")
    # print("gathered:")
    # print(gathered)
    
    tmp_final_distance = [[gathered[i][j][0] for j in range(len(gathered[i]))] for i in range(len(gathered))]        
    tmp_final_indices = [[gathered[i][j][1] for j in range(len(gathered[i]))] for i in range(len(gathered))]   
    final_distance, final_indices = [[]], [[]]
    for t in tmp_final_distance:
        final_distance[0].extend(list(t[0])) 
    for t in tmp_final_indices:
        final_indices[0].extend(list(t[0])) 
    # print(f"len(final_distance):{len(final_distance)}")
    # print(f"len(final_distance[0]):{len(final_distance[0])}")
    # print(f"len(final_indices):{len(final_indices)}")
    # print(f"len(final_indices[0]):{len(final_indices[0])}")
    if not single_mode:
        return final_distance, final_indices

    else:
        return final_distance[0], final_indices[0]