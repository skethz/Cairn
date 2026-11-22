use tch::{IndexOp, Kind, Tensor};
pub fn chunk_delta(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    log_decay: &Tensor,
    beta: &Tensor,
    chunk: i64,
    initial: Option<&Tensor>,
) -> (Tensor, Tensor) {
    assert!(chunk > 0 && q.size()[2] > 0);
    crate::native::autocast(false, || {
        let length = q.size()[2];
        let kind = if q.kind() == Kind::Double {
            Kind::Double
        } else {
            Kind::Float
        };
        let mut q = q.to_kind(kind);
        let mut k = k.to_kind(kind);
        let mut v = v.to_kind(kind);
        let mut decay = log_decay.to_kind(kind);
        let mut beta = beta.to_kind(kind);
        let pad = (chunk - length % chunk) % chunk;
        if pad > 0 {
            q = q.constant_pad_nd([0, 0, 0, pad]);
            k = k.constant_pad_nd([0, 0, 0, pad]);
            v = v.constant_pad_nd([0, 0, 0, pad]);
            decay = decay.constant_pad_nd([0, 0, 0, pad]);
            beta = beta.constant_pad_nd([0, pad]);
        }
        let (b, h, t, dk) = (q.size()[0], q.size()[1], q.size()[2], q.size()[3]);
        let n = t / chunk;
        let dv = v.size()[3];
        let q = q.reshape([b, h, n, chunk, dk]);
        let k = k.reshape([b, h, n, chunk, dk]);
        let v = v.reshape([b, h, n, chunk, dv]);
        let g = decay
            .reshape([b, h, n, chunk, decay.size()[3]])
            .cumsum(-2, None);
        let beta = beta.reshape([b, h, n, chunk, 1]);
        let future = Tensor::ones([chunk, chunk], (Kind::Bool, q.device())).triu(1);
        let (kk, qk) = if g.size()[4] == 1 {
            let d = (&g - g.transpose(-1, -2))
                .masked_fill(&future, 0.)
                .exp()
                .masked_fill(&future, 0.);
            (
                k.matmul(&k.transpose(-1, -2)) * &d,
                q.matmul(&k.transpose(-1, -2)) * d,
            )
        } else {
            let d = (g.unsqueeze(-2) - g.unsqueeze(-3))
                .masked_fill(&future.unsqueeze(-1), 0.)
                .exp()
                .masked_fill(&future.unsqueeze(-1), 0.);
            (
                (k.unsqueeze(-2) * k.unsqueeze(-3) * &d).sum_dim_intlist(
                    [-1].as_slice(),
                    false,
                    None,
                ),
                (q.unsqueeze(-2) * k.unsqueeze(-3) * d).sum_dim_intlist(
                    [-1].as_slice(),
                    false,
                    None,
                ),
            )
        };
        let lower = Tensor::eye(chunk, (kind, q.device())) + (&beta * &kk).tril(-1);
        let w = lower.linalg_solve_triangular(&(&beta * &v), false, true, true);
        let u = lower.linalg_solve_triangular(&(&beta * &k * g.exp()), false, true, true);
        let read = &q * g.exp() - qk.matmul(&u);
        let local = qk.matmul(&w);
        let tail = &k * (g.narrow(-2, chunk - 1, 1) - &g).exp();
        let transition = g
            .select(-2, chunk - 1)
            .expand([b, h, n, dk], false)
            .exp()
            .diag_embed(0, -2, -1)
            - tail.transpose(-1, -2).matmul(&u);
        let write = tail.transpose(-1, -2).matmul(&w);
        let mut state = initial
            .map(|x| x.to_kind(kind))
            .unwrap_or_else(|| Tensor::zeros([b, h, dk, dv], (kind, q.device())));
        if crate::native::scan_enabled() {
            let (out,state)=crate::native::scan(&read,&local,&transition,&write,&state).expect("Native chunk VJP");
            return (out.reshape([b,h,t,dv]).narrow(2,0,length),state);
        }
        let mut outputs = Vec::new();
        for c in 0..n {
            outputs.push(local.i((.., .., c)) + read.i((.., .., c)).matmul(&state));
            state = transition.i((.., .., c)).matmul(&state) + write.i((.., .., c));
        }
        (
            Tensor::stack(&outputs, 2)
                .reshape([b, h, t, dv])
                .narrow(2, 0, length),
            state,
        )
    })
}
