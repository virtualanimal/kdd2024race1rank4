## Xgboost  

### Introduction：

Do feature engineering and train xgboost model to predict results at 10 fold

### Feature Description

```
total_w_cotitle: 同名作者的一篇论文与其他论文的共同title词权重之和
total_w_coauthor: 同名作者的一篇论文与其他论文的共同author词权重之和
total_w_coorg: 同名作者的一篇论文与其他论文的共同org词权重之和
total_w_coabstract: 同名作者的一篇论文与其他论文的共同abstract词权重之和
total_w_cokeywords: 同名作者的一篇论文与其他论文的共同keywords词权重之和
total_w_covenue: 同名作者的一篇论文与其他论文的共同venue词权重之和
co_auther: 同名作者的一篇论文与其他论文的共同auther词权重的平均值（不同计算方法）
co_auther_org: 同名作者的一篇论文与其他论文的共同org词权重的平均值（不同计算方法）
co_venue: 同名作者的一篇论文与其他论文的共同venue词权重的平均值（不同计算方法）
co_keywords: 同名作者的一篇论文与其他论文的共同keywords词权重的平均值（不同计算方法）
total_title_bge_sim: 同名作者的一篇论文与其他论文的title的bge嵌入后计算相似度之和
total_abstract_bge_sim: 同名作者的一篇论文与其他论文的abstract的bge嵌入后计算相似度之和
total_title_and_abstract_bge_sim: 同名作者的一篇论文与其他论文的title和abstract结合的bge嵌入后计算相似度之和
total_keyword_bge_sim: 同名作者的一篇论文与其他论文的keywords的bge嵌入后计算相似度之和
total_venue_bge_sim: 同名作者的一篇论文与其他论文的venue的bge嵌入后计算相似度之和
total_title_scibert_sim: 同名作者的一篇论文与其他论文的title的scibert嵌入后计算相似度之和
total_abstract_scibert_sim: 同名作者的一篇论文与其他论文的abstract的scibert嵌入后计算相似度之和
total_title_and_abstract_scibert_sim: 同名作者的一篇论文与其他论文的title和abstract结合的scibert嵌入后计算相似度之和
total_keyword_scibert_sim: 同名作者的一篇论文与其他论文的keyword的scibert嵌入后计算相似度之和
total_venue_scibert_sim: 同名作者的一篇论文与其他论文的venue的scibert嵌入后计算相似度之和
total_oagbert_sim: 同名作者的一篇论文与其他论文oagbert嵌入后计算相似度之和
title_len: 论文title长度
abstract_len: 论文abstract长度
keywords_len: 论文keywords列表长度
authors_len: 论文authors长度
venue_len: 论文venue长度
year: 论文出版年份
title_vec: 论文title的bge嵌入向量
abstract_vec: 论文abstract的bge嵌入向量
title_and_abstract_vec：论文title和abstract结合之后的bge嵌入向量
keywords_vec: 论文的keywords的bge嵌入向量
venue_vec: 论文venue的bge嵌入向量
paper_oagbert_vec: 整篇论文oagbert嵌入向量
```