from d2l import torch as d2l

valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
 valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)