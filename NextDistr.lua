
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'

local NextDistr = {}
NextDistr.__index = NextDistr

function NextDistr.create()
    local self = {}
    setmetatable(self, NextDistr)
    
    fetch_events()
    print('There are ' .. opt.num_events .. ' unique events in redis')
    
    collectgarbage()
    return self
end

function NextDistr:create_rnn_units_and_criterion()
  if string.len(opt.init_from) == 0 then
    print('creating an model with ' .. opt.layer_sizes .. ' layers')
    protos = {}
    if opt.rnn_unit == 'lstm' then
      protos.rnn = LSTM.lstm()
    elseif opt.rnn_unit == 'gru' then
      protos.rnn = GRU.gru()
    end
    
    protos.criterion = nn.MSECriterion()
  end
end


function NextDistr:next_batch()
  collectgarbage()
  
  local seqs = loadBatch(opt.batch_size, opt.len_seq)
  --print ("seqs", seqs)
  --local num_events = #seqs[1]:split(",")
  
  local num_obs = seqs[1]["distributions"]:size()[1]
  
  local t_x = torch.DoubleTensor(num_obs, #seqs, theta_size):zero()
  local d_x = torch.DoubleTensor(num_obs, #seqs, opt.num_events):zero()
  
  
  for s=1,#seqs do
    t_x:sub(1,-1,s,s):copy(seqs[s]["timestamps"])
    d_x:sub(1,-1,s,s):copy(seqs[s]["distributions"])
  end
    
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    t_x = t_x:float():cuda()
    d_x = d_x:float():cuda()
  end
  return t_x, d_x
end

function NextDistr:feval()
    grad_params:zero()

    ------------------ get minibatch -------------------
    local t_x, d_x = opt.loader:next_batch()
    
    local len_seq = opt.len_seq
    
    local rnn_state = {[0] = init_state_global}
    local predictions = {}
    local loss = 0
    local last_max
    local init_state = init_state()

    for t=1,len_seq do
       if not (batch_type == "training") then
        clones.rnn[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
      else
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      end
      
      local lst = clones.rnn[t]:forward{t_x[t], d_x[t], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      predictions[t] = lst[#lst] --
      
      loss = loss + clones.criterion[t]:forward(predictions[t], d_x[t+1])
    end
      
    loss = loss / len_seq
    
    if not (batch_type == "training") then
      return loss, grad_params
    end
      
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[len_seq] = clone_list(init_state, true)} -- true also zeros the clones
    for t=len_seq,1,-1 do
      -- backprop through loss, and softmax/linear
      
      local doutput_t = clones.criterion[t]:backward(predictions[t], d_x[t+1])
      table.insert(drnn_state[t], doutput_t)
      
      local dlst = clones.rnn[t]:backward({d_x[t+1], unpack(rnn_state[t-1])}, drnn_state[t])
      
      drnn_state[t-1] = {}
      
      for k,v in pairs(dlst) do
        if k > 2 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          drnn_state[t-1][k-2] = v
        end
      end
    end
    
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    grad_params:div(len_seq) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    
    return loss, grad_params
end


return NextDistr

