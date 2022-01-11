'''
Model the distribution of the latent factors giving rise to an image. Can
sample latent factors, determine their likelihoods and likelihood ratios, and
so on.
'''

class BaseGenerator:
    '''Model to generate the true latent factors.'''

    def __init__(self, renderer, **kwargs):
        '''
        Constructor arguments:
        renderer (callable):
            given a set keyword parameters specified here, the
            renderer should return the rendered output.
        keyword arguments (subclass of latent_model.distribution.Distribution):
            the keyword arguments should correspond to the arguments of the
            renderer. The arguments for the renderer by default will be sampled
            from the respective distribution given here.
        '''
        self.renderer = renderer

        if kwargs:
            self.distributions = {'default': kwargs}
        else:
            self.distributions = {}

    def set_distribution(self, name, inherit_from=None, **kwargs):
        '''
        Define a new, named, distribution of latent factors.
        Arguments:
        name (string):
            Name for the new distribution.
        inherit_from (optional, string):
            Use an existing distribution (e.g. 'default') as a starting point,
            only change the distribution for some of the kwargs.
        keyword arguments (subclass of latent_model.distribution.Distribution):
            The keyword arguments should correspond to the arguments of the
            renderer. The arguments for the renderer will be sampled from the
            respective distribution given here.
        Returns:
        nothing
        '''

        if inherit_from:
            self.distributions[name] = self.distributions[inherit_from].copy()
            self.distributions[name].update(kwargs)
        else:
            self.distributions[name] = kwargs

    def log_likelihood(self, distribution='default', **kwargs):
        '''
        Evaluate the log likelihood of a given set of kwargs under one of the
        latent factor distributions.
        Arguments:
        distribution (optional, string):
            the name of the latent factor distribution under which to evaluate
            the log likelihood. By default uses the 'default' distribution.
        keyword arguments (type depends on renderer):
            the values for which to compute the log likelihood. Unspecified
            keywords known to the latent model are marginalized out.
        Returns:
        a single float
        '''
        log_p = 0.
        dist = self.distributions[distribution]

        for k, v in kwargs.items():
            log_p += dist[k].log_likelihood(v)

        return log_p

    def sample(self, n=None, distribution='default', **kwargs):
        '''
        Sample parameters from the model of latent factors.
        Arguments:
        n (optional, int or None):
            how many samples to generate. If None, return a single sample directly
            (as opposed to a list with only one entry, for n=1)
        distribution (optional, string):
            Which latent distribution to sample from. Uses 'default' by default.
        kwargs (optional, keyword arguments):
            The keyword arguments to the renderer.
        Returns:
        list of tuples of the form (render output, used kwargs),
        or the tuple directly for n=None
        '''

        dist = self.distributions[distribution]
        sampled_input_properties = {k: g.sample(n or 1) for k, g in dist.items()}
        outputs = []

        for k in range(n or 1):
            input_properties_k = {p: v[k] for p, v in sampled_input_properties.items()}
            outputs.append((self.renderer(**{**input_properties_k, **kwargs}), input_properties_k))

        return (outputs if n else outputs[0])
