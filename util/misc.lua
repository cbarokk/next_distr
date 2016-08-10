
-- misc utilities
require 'gnuplot'

local redis = require 'redis'
redis_client = redis.connect('127.0.0.1', 6379)


function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function Welch(N)
  local w = torch.Tensor(N)
  local i = -1
  local half = (N-1)/2
  w:apply(function()
    i = i + 1   
    return 1-(math.pow((i-half)/half,2))
  end)
  w:div(w:sum())
  return w
end

local welch_table = {}

function smooth_probs(probs, N)
  if welch_table[N] == nil then
    local welch_batch = torch.DoubleTensor(probs:size()[1], N*2+1)
    local w = Welch(2*N+1)
    
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
      w = w:float():cuda()
      welch_batch = welch_batch:float():cuda()
    end
    for i=1, probs:size()[1] do
      welch_batch:sub(i,i):copy(w)
    end
    welch_table[N] = welch_batch
  end
  
  local w = welch_table[N]
  --local left_half_w = w:sub(1,N/2+1):div(w:sub(1,N/2+1):sum())
  
  local tmp = torch.cat({
      probs:sub(1,-1, probs:size()[2] - N+1, -1), 
      probs,
      probs:sub(1,-1, 1, N)},2)
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    tmp = tmp:float():cuda()
  end
  
  local offset = N
    for i = 1, probs:size()[2] do
      probs:sub(1,-1,i,i):fill(torch.cmul(tmp:sub(1,-1,offset+i-N, offset+i+N),w):sum())
    end
  probs:div(probs:sum()) -- normalize so it sums to 1
end
  

theta_size = 8

function timestamp2theta(timestamp)
  local theta = torch.DoubleTensor(theta_size):fill(0)
  
  local date = os.date("*t", timestamp)
  
  
  local sec = date['sec']
  theta[1] = math.cos((2*math.pi)/60*sec) --cos_sec
  theta[2] = math.sin((2*math.pi)/60*sec) --sin_sec
  
  local min = date['min']
  theta[3] = math.cos((2*math.pi)/60*min) --cos_min
  theta[4] = math.sin((2*math.pi)/60*min) --sin_min
      
  local hour = date['hour']
  theta[5] = math.cos((2*math.pi)/24*hour) --cos_hour
  theta[6] = math.sin((2*math.pi)/24*hour) --sin_hour
      
  local weekday = date['wday']-1
  theta[7] = math.cos((2*math.pi)/7*weekday) --cos_weekday
  theta[8] = math.sin((2*math.pi)/7*weekday) --sin_weekday
  --[[
  local monthday = date['day']
  theta[9] = math.cos((2*math.pi)/31*monthday) --cos_monthday
  theta[10] = math.sin((2*math.pi)/31*monthday) --sin_monthday

  local month = date['month']
  theta[11] = math.cos((2*math.pi)/12*month) --cos_month
  theta[12] = math.sin((2*math.pi)/12*month) --sin_month
  
  local yearday = date['yday']
  theta[13] = math.cos((2*math.pi)/365*yearday) --cos_yearday
  theta[14] = math.sin((2*math.pi)/365*yearday) --sin_yearday
]]--
  return theta, date
end

    
  function fetch_events()
    if string.len(opt.init_from) > 0 then --
      print "recovering options from checkpoint"
      opt.num_events = checkpoint.opt.num_events
      opt.event_mapping = checkpoint.opt.event_mapping
      opt.event_inv_mapping = checkpoint.opt.event_inv_mapping
    else
      opt.num_events = redis_client:scard(opt.redis_prefix .. '-events')
      local tmp = redis_client:smembers(opt.redis_prefix .. '-events')
      opt.num_events = # tmp
      opt.event_mapping = {}
      opt.event_inv_mapping = {}

      for id, event_id in pairs(tmp) do
        opt.event_mapping[event_id] = id
        opt.event_inv_mapping[id+1] = event_id
      end
    end

end

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

  
function init_state()
  local state={}
  for L=1, #opt.rnn_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_layers[L])
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(state, h_init:clone())
    if opt.rnn_unit == 'lstm' then
        table.insert(state, h_init:clone())
    end
  end
  return state
end

local function gpu_init()
   -- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
  if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if not (ok and ok2) then
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end
end


function recover_opt_from_checkpoint()
  
  opt.do_random_init = true
  if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos

    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting options from checkpoint.')
    print ("last learning rate", checkpoint.learning_rate)
    opt.rnn_layers = checkpoint.opt.rnn_layers
    opt.rnn_unit = checkpoint.opt.rnn_unit
    opt.seed = checkpoint.opt.seed
    opt.start_epoch = checkpoint.epoch
    opt.do_random_init = false
  else
    opt.rnn_layers = {unpack(opt.layer_sizes:split(","))}
    for i =1, #opt.rnn_layers do
      opt.rnn_layers[i] = tonumber(opt.rnn_layers[i])
    end
    opt.start_epoch = 0
  end
  
  torch.manualSeed(opt.seed)

  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  end
end


function lstm_init()
  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if opt.rnn_unit == 'lstm' then
    for layer_idx = 1, #opt.rnn_layers do
      for _,node in ipairs(protos.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
          print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
          -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
          node.data.module.bias[{{opt.rnn_layers[layer_idx]+1, 2*opt.rnn_layers[layer_idx]}}]:fill(1.0)
        end
      end
    end
  end
  
  -- ship the model to the GPU if desired
  if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
  end
end

function init()
  gpu_init()
  recover_opt_from_checkpoint()
  donkey_init()
end
  