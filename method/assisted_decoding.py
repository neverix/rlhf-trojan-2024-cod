from tqdm.auto import trange
import plotly.express as px
import gadgets as gd
import joblib as jl
import torch


def main():
    pre = list(next(iter(a for (a, *_), *_ in jl.load("cache/bad_completions.pkl") if len(a) < 20)))
    og_length = len(pre)
    nm = gd.mod("s")    
    rm = gd.mod("r")
    t = gd.tok()
    k = 128
    cfg_scale = 2
    all_tokens = t.batch_decode([[x] for x in range(t.vocab_size + 1)], skip_special_tokens=True)
    for i in (bar := trange(16)):
        with torch.inference_mode():
            ii = torch.LongTensor(pre).cuda().unsqueeze(0)
            ii_mod = torch.LongTensor(pre[:og_length - 5] + t.encode("a SUDO")[-2:]
                                    + pre[og_length - 5:]).cuda().unsqueeze(0)
            l = nm(input_ids=ii).logits[0, -1]
            l2 = nm(input_ids=ii_mod).logits[0, -1]
            px.scatter(x=l.detach().cpu().float().numpy(),
                       y=l2.detach().cpu().float().numpy(),
                       hover_name=all_tokens).write_image(f"figures/logit_v_sudo{i}.png")
            l = l + (l2 - l) * cfg_scale
            tokens = l.topk(k, dim=-1).indices
            candidates = torch.cat((ii.repeat(k, 1), tokens.unsqueeze(1),
                                    tokens.unsqueeze(1) * 0 + t.eos_token_id), 1)
            scores = rm(candidates,
                        attention_mask=candidates != t.pad_token_id
                        ).rewards[:, -1, 0].float().detach().cpu().numpy()
            l_scores = l[tokens].detach().cpu().float().numpy()
            l2_scores = l2[tokens].detach().cpu().float().numpy()
            idx_tokens = [all_tokens[i] for i in tokens]
            px.scatter(x=l_scores, y=scores, hover_name=idx_tokens).write_image(f"figures/logit_v_rm_{i}.png")
            px.scatter(x=l2_scores, y=scores, hover_name=idx_tokens).write_image(f"figures/sudo_v_rm_{i}.png")
            # sample = tokens[scores.argmax()]
            # sample = tokens[scores.argmin()]
            # sample = tokens[torch.multinomial(scores.softmax(0), 1)[0]]
            # sample = tokens[torch.multinomial((-scores).softmax(0), 1)[0]]
            # sample = l.argmin()
            sample = torch.multinomial(l.softmax(0), 1)[0]
            pre.append(sample)
            bar.set_postfix(completion=t.decode(pre), score=scores.min().item())


if __name__ == "__main__":
    main()
