# Result

## A. Total Number of Source Titles / Tokenized Titles

| Item | Count |
| --- | --- |
| Total Number of Source Titles | 793,612 |
| Total Number of Tokenized Titles | 577,600 |

## B. If A and B are different, what have you done for that?

1. 在 `cleanFile` 的時候，有把不必要的資訊砍掉，導致整行被砍掉，例如 `[]` 或是 `Re:` 等等。
2. 再搭配 CKIP 取得斷詞後，又有把一些不必要的詞砍掉，例如連接詞等等，有些行句斷詞 filter 之後會導致整行砍掉。
3. 刪除大量重複資訊（刪除最大宗）。

## C. Parameters of Doc2Vec Embedding Model

| Parameter | Value |
| --- | --- |
| Total Number of Training Documents | 577,600 |
| Output Vector Size | 40 |
| Min Count | default |
| Epochs | 100 |
| Workers | default |
| First Self Similarity | 81.8% |
| Second Self Similarity | 86.9% |

## D. Parameters of Multi-Class Classification Model

| Parameter | Value |
| --- | --- |
| Arrangement of Linear Layers | 40 × 100 × 50 × 9 |
| Activation Function for Hidden Layers | ReLU |
| Activation Function for Output Layer | Softmax |
| Loss Function | Categorical Cross Entropy |
| Algorithm for Back-Propagation | Adam |
| Total Number of Training Documents | 462,080 |
| Total Number of Testing Documents | 115,520 |
| Epochs | 300 |
| Learning Rate | 0.001 |
| Accuracy on Testing Documents | 81.32% |

## E. Share your experience of optimization, including at least 2 change/result pairs

| # | Change | Result |
| --- | --- | --- |
| 1 | 層數由 `40*60*50*9` 改成 `40*100*50*9` | 正確率 80% → 81% |
| 2 | batch size 由 256 改成 32 | 正確率沒有任何變化 |
| 3 | 學習率由 0.01 改成 0.001 | 正確率沒有任何變化 |
