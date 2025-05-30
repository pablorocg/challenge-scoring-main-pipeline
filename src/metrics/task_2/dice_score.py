from sklearn.metrics import f1_score

dsc = f1_score(y_true_mask.flatten(), y_pred_mask.flatten())
