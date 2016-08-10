--[[

This file trains a event-based multi-layer RNN on timestamped data

Code is based on char-rnn from karpathy based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'

require 'donkey'
require 'util.misc'
local model_utils = require 'util.model_utils'

local NextDistr = require 'NextDistr'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a timestamped events model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-rnn_unit', 'lstm', 'lstm,gru or rnn')
cmd:option('-layer_sizes', '256', 'size of the stacked rnn layers')
cmd:option('-batch_size', 50,'number of sequences per batch')
cmd:option('-len_seq', 30,'number of events per sequence')
cmd:option('-data', 'data', 'path to data when sequence source is disk')

-- optimization
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-patience',5,'number of epochs, before considering decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-max_epochs',5000,'number of full passes through the training data')
cmd:option('-iterations_per_epoch',25,'number of iterations per epoch')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-test_every', 5,'how many epochs between testing an epoch and dumping a checkpoint')

cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpoint to. Will be inside checkpoint_dir/')
cmd:option('-redis_prefix', 'next_distr', 'redis key name prefix, where to read train/validation/events data')
cmd:option('-info', '', 'small string, just to simplify viewing several plotting windows at once')

-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

init()

opt.loader = NextDistr.create()
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
opt.loader:create_rnn_units_and_criterion()

lstm_init()

--[[
  -- Visualzing the net
  nngraph.setDebug(true)
  protos.rnn.name = 'cyril_rnn_net'
  local input = torch.rand(11)
  pcall(function() protos.rnn:updateOutput(input) end)
  os.execute('open -a Safari cyril_rnn_net.svg')
  sys.sleep(10000)
 ]]--
 

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if opt.do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

print('number of parameters in the model: ' .. params:nElement())
 
 -- define the model: prototypes for one timestep, then clone them in time
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.len_seq, not proto.parameters)
end

init_state_global = clone_list(init_state())

-- start optimization here
local train_losses = {}
local test_losses = {}
local learning_rates={}

local loss0 = nil
local accum_loss

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local decay_learning_rate = false

for epoch = opt.start_epoch, opt.max_epochs do
  accum_loss = 0
  batch_type = "training"
  
  if epoch % opt.test_every == 0 then
    print ("testing an epoch")
    batch_type = "validation"
  end

  for i = 1, opt.iterations_per_epoch do
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(opt.loader.feval, params, optim_state)
    --local _,loss = optim.adagrad(opt.loader.feval, params, optim_state)
    --local _,loss = optim.sgd(opt.loader.feval, params, optim_state)
    --local _,loss = optim.adam(opt.loader.feval, params, optim_state)
    
    local time = timer:time().real
    local batch_loss = loss[1] -- the loss is inside a list, pop it
    
    accum_loss = accum_loss + batch_loss
  
    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, opt.iterations_per_epoch, epoch, batch_loss, grad_params:norm() / params:norm(), time))
    end
     
     if batch_type == "training" then
      -- handle early stopping if things are going really bad
      if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
      end
    end
  end
    
  accum_loss = accum_loss / opt.iterations_per_epoch
  if batch_type == "training" then
    train_losses[#train_losses+1] = accum_loss
    gnuplot.figure('train losses')
    gnuplot.title('train losses ' .. opt.info)
    gnuplot.plot({torch.Tensor(train_losses),'-'})
  else
    test_losses[#test_losses+1] = accum_loss
    gnuplot.figure('test losses')
    gnuplot.title('test losses ' .. opt.info)
    gnuplot.plot({torch.Tensor(test_losses),'-'})
    
    local _, min_idx = torch.Tensor(test_losses):min(1)
    if #test_losses - min_idx[1] > opt.patience then
      local decay_factor = opt.learning_rate_decay
      optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
      print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    end
    
    -- dump a checkpoint
    local savefile = string.format('%s/model_%s_%s_epoch%.2f_%.8f.t7', opt.checkpoint_dir, opt.rnn_unit,         opt.layer_sizes, epoch, accum_loss)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    
    for _,node in ipairs(protos.rnn.forwardnodes) do
      if node.data.annotations.name == "bn_i2h_1" then
        print ("rnn: running_mean:", node.data.module.running_mean:sub(1,10))
      end
    end
    
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.test_losses = test_losses
    checkpoint.epoch = epoch
    checkpoint.learning_rate = optim_state.learningRate
    torch.save(savefile, checkpoint)

  end
  
  learning_rates[#learning_rates+1] = optim_state.learningRate
  gnuplot.figure('learning rate')
  gnuplot.title('learning rate ' .. opt.info)
  gnuplot.plot({torch.Tensor(learning_rates),'-'})
  
  
  if loss0 == nil then loss0 = accum_loss end
  if accum_loss > loss0 * 3 then
    print('loss is exploding, aborting.')
    break -- halt
  end
  
end


