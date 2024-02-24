from prompt_search import make_judger, parse_judgement_type
from tqdm.auto import tqdm, trange
import gadgets as gd
import torch


def main(judgement_type: str = "logprob-2-1x32x4-rt--4",
         seed: int = 0, n_special: int = 8, lr: float = 3e-2,
         dist_loss_weight: float = 10, random_scale: float = 0.01,
         inner_epochs: int = 100, outer_epochs: int = 100):
    torch.manual_seed(seed)
    
    tokenizer = gd.tok()
    judger = make_judger(judgement_type)
    next(judger)
    judger.send(True)
    name, *_ = parse_judgement_type(judgement_type)
    model = gd.mod(name)
    w = model.model.embed_tokens.weight
    
    def soft_quantize(embed):
        prob_dist = ((w.unsqueeze(-1) * embed.T.unsqueeze(0)).sum(dim=1)).softmax(dim=0).T
        return prob_dist.to(w) @ w
    
    def quantize_embed(embed):
        z = (w.unsqueeze(-1) - embed.T.unsqueeze(0)).norm(dim=1).argmin(dim=0)
        return z, model.model.embed_tokens(z)
    
    def z_q(embed):
        _, z = quantize_embed(embed)
        return (embed - embed.detach()) + z.detach()

    def token_dist(embed):
        return (w.unsqueeze(-1) - embed.T.unsqueeze(0)).norm(dim=1).min(dim=0).values.mean()
    
    special_embeds = torch.nn.Parameter(torch.randn(n_special, model.config.hidden_size).cuda(),
                                        requires_grad=True)
    optimizer = torch.optim.Adam([special_embeds], lr=lr)
    best_prob, best_tokens = -float("inf"), None
    for i in (bar := trange(outer_epochs)):
        for _ in (minibar := trange(inner_epochs)):
            embeds = special_embeds
            # embeds = soft_quantize(special_embeds)
            # embeds = z_q(special_embeds)
            judger.send(embeds)
            prob, = next(judger)
            loss = -prob + token_dist(special_embeds) * dist_loss_weight
            minibar.set_postfix(prob=prob.item(), loss=loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            tokens, _ = quantize_embed(special_embeds)
            judger.send(False)
            judger.send(tokens.tolist())
            prob = next(judger)[0]
            judger.send(True)
            if prob > best_prob:
                best_prob, best_tokens = prob, tokens
            bar.set_postfix(prob=prob, tokens=tokens.tolist(), prompt=tokenizer.decode(tokens))
            special_embeds[:] = model.model.embed_tokens(best_tokens)
            special_embeds += torch.randn_like(special_embeds) * special_embeds.std() * random_scale
            optimizer = torch.optim.Adam([special_embeds], lr=lr)
