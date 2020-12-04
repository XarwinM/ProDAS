class BaseGenerator:

    def __init__(self, renderer, **kwargs):

        self.renderer = renderer

        if kwargs:
            self.distributions = {'default': kwargs}
        else:
            self.distributions = {}

    def set_distribution(self, name, **kwargs):
        self.distributions[name] = kwargs

    def log_likelihood(self, distribution='default', **kwargs):
        log_p = 0.
        dist = self.distributions[distribution]

        for k,v in kwargs.items():
            log_p += dist[k].log_likelihood(v)

        return log_p

    def sample(self, n=None, distribution='default'):

        # TODO: also output the sampled parameters (duh!)

        dist = self.distributions[distribution]
        sampled_input_properties = {k:g.sample(n or 1) for k,g in dist.items()}
        outputs = []

        for k in range(n or 1):
            input_properties_k = {p: v[k] for p,v in sampled_input_properties.items()}
            outputs.append((self.renderer(**input_properties_k), input_properties_k))

        return (outputs if n else outputs[0])



if __name__ == '__main__':

    import distributions as pd
    import pprint

    def _debug_renderer(**kwargs):
        pprint.pprint(kwargs, width=30)
        print()
        return 0.

    gen = BaseGenerator(_debug_renderer,
                        color     = pd.DiscreteChoice(choices=[0,1,2]),
                        thickness = pd.ContinuousUniform(0.1, 0.9),
                        boucyness = pd.ContinuousNormal(50., 25.))

    gen.set_distribution('ood_1',
                         color     = pd.DiscreteChoice(choices=[0,1,2]),
                         thickness = pd.ContinuousUniform(1.0, 0.2),
                         boucyness = pd.DiscreteChoice(choices=[100.]))

    print('single iD sample:')
    gen.sample()

    print('3 iD samples:')
    gen.sample(n=3)

    print('3 OoD samples:')
    gen.sample(n=3, distribution='ood_1')

    test_inputs = [{'color': 1, 'thickness': 0.3, 'boucyness': 62.5},
                   {'color': 2, 'thickness': 0.7, 'boucyness': 190.87},
                   {'color': 4, 'thickness': 0.4, 'boucyness': 46.2}]

    for t in test_inputs:
        print(f'log-likelihood of {t}:')
        print(gen.log_likelihood(**t))
