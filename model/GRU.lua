
local GRU_theta = {}

--[[
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU_theta.gru()
  local rnn_size = opt.rnn_size
  local num_time_slots = opt.num_time_slots
  local num_layers = opt.num_layers
  local num_eoi = opt.num_eoi
  local num_events = opt.num_events
  local dropout = opt.dropout or 0
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- e_x
  for L = 1, num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x, input_size_L
  local outputs = {}
  for L = 1, num_layers do

    local prev_h = inputs[L+2]
    -- the input to this layer

    if L == 1 then 
      local theta_x = inputs[1]
      local e_x = inputs[2]
      local embedings_size = 100
      local e_embedings = nn.LookupTable(num_events, embedings_size)(e_x):annotate{name='emb_e'}
      e_embedings = nn.Reshape(embedings_size)(e_embedings)
      
      --x = nn.JoinTable(2)({theta_embedings, e_embedings}) 
      x = nn.JoinTable(2)({theta_x, e_embedings}) 
      input_size_L = theta_size + embedings_size
      --x = OneHot(input_size)(inputs[1])
      --input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    
    x = nn.BatchNormalization(input_size_L)(x)
    
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
  local layer = outputs[#outputs]
  
  if dropout > 0 then layer = nn.Dropout(dropout)(layer) end
  
  local time_pred = nn.Linear(rnn_size, num_time_slots)(layer)
  time_pred = nn.LogSoftMax()(time_pred)
  table.insert(outputs, time_pred)
  
  local event_pred = nn.Linear(rnn_size, num_eoi)(layer)
  event_pred = nn.LogSoftMax()(event_pred)
  table.insert(outputs, event_pred)
  
  
  
  return nn.gModule(inputs, outputs)
end

return GRU_theta
