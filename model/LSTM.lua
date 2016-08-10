--require 'util.MyDropout'

local LSTM_NextEvent = {}

function LSTM_NextEvent.lstm()
  local num_events = opt.num_events
  local dropout = opt.dropout or 0
  local rnn_layers = opt.rnn_layers
  
  -- there will be 2*n+2 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- t_x
  table.insert(inputs, nn.Identity()()) -- d_x
  
  local event_embed_table

    function LSTM_Module(input_size, input, prev_c, prev_h, output_size, annotation)
      -- evaluate the input sums at once for efficiency
      local i2h = nn.Linear(input_size, 4 * output_size)(input):annotate{name='i2h_'..annotation}
      local h2h = nn.Linear(output_size, 4 * output_size)(prev_h):annotate{name='h2h_'..annotation}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      all_input_sums = nn.BatchNormalization(4*output_size)(all_input_sums)
      
      local reshaped = nn.Reshape(4, output_size)(all_input_sums)
      local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
      -- decode the gates
      local in_gate = nn.Sigmoid()(n1):annotate{name='in_gate_'..annotation}
      local forget_gate = nn.Sigmoid()(n2):annotate{name='forget_gate_'..annotation}
      local out_gate = nn.Sigmoid()(n3):annotate{name='out_gate_'..annotation}
      -- decode the write inputs
      local in_transform = nn.Tanh()(n4)
      -- perform the LSTM update
      local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      }):annotate{name='next_c_'..annotation}
      -- gated cells form the output
      
      local next_c_h = nn.BatchNormalization(output_size)(next_c)
      
      local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c_h)}):annotate{name='next_h_'..annotation}
      return next_c, next_h
    end


  for L = 1,#rnn_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local input_t_x, input_ex, input_size_t_x, input_size_d_x
  local outputs = {}
  local embedings_size = 100
      
  for L = 1,#rnn_layers do

  -- c,h from previos timesteps
    local prev_c = inputs[L*2+1]
    local prev_h = inputs[L*2+2]
    
    -- the input to this layer
    if L == 1 then 
      input_t_x = inputs[1]
      input_d_x = inputs[2]
      
      input_x = nn.JoinTable(2)({input_t_x, input_d_x})
      
      input_size_x = theta_size + num_events
      
    else 
      input_x = next_h_x
      
      --if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_x = rnn_layers[L-1]/2
      
    end

    if dropout > 0 then input_x = nn.Dropout(dropout)(input_x) end -- apply dropout, if any

    local next_c, next_h = LSTM_Module(input_size_x, input_x, prev_c, prev_h, rnn_layers[L], L)

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  
  local layer = outputs[#outputs]

  if dropout > 0 then layer = nn.Dropout(dropout)(layer) end -- apply dropout, if any

  local layer_size = rnn_layers[#rnn_layers]
  
  local proj = nn.Linear(layer_size, num_events)(layer):annotate{name='softmax events'}
  --local logsoft = nn.LogSoftMax()(proj)
  --table.insert(outputs, logsoft)
  table.insert(outputs, proj)
  
  return nn.gModule(inputs, outputs)
end

return LSTM_NextEvent

