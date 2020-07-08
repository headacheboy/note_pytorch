torch.nn.utils.clip_grad_norm_(model.parameters(), clip_num)

写在loss.backward()之后