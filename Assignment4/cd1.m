function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

    % Addition to deal with non-binary input data
    visible_data = sample_bernoulli(visible_data);
    
    hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);
    hidden_probabilities = sample_bernoulli(hidden_probabilities);
    
    positive_gradient = configuration_goodness_gradient(visible_data, hidden_probabilities);
    
    visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_probabilities);
    visible_probabilities = sample_bernoulli(visible_probabilities);
    
    hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_probabilities);
    % hidden_probabilities = sample_bernoulli(hidden_probabilities);
    
    negative_gradient = configuration_goodness_gradient(visible_probabilities, hidden_probabilities);
    
    ret = positive_gradient - negative_gradient;
end
