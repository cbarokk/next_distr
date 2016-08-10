require 'lfs'
require 'xlua' 

local BUFSIZE = 2^14     -- 16K
local maximum = 0
local num_var
local raw_data
local data_split = {["training"]={["start"]=0, ["stop"]=0},
                    ["validation"]={["start"]=0, ["stop"]=0},
                    ["testing"] = {["start"]=0, ["stop"]=0},}

local function loadChunk(file, data, bufsize, seek)
  if seek then file:seek("set", seek) end
  local buf_size = buf_size or BUFSIZE
  local buf, rest = file:read(buf_size, "*line")
  if not buf then return false end
  if rest then buf = buf .. rest .. '\n' end
  table.insert(data, buf)
  return true
end

local function parseLine(chunk, lines)
  for line in chunk:gmatch("[^\r\n]+") do 
    table.insert(lines, line)
  end
end
  

local function parseLines(data)
  local lines={}
  for _, chunk in pairs(data) do
    parseLine(chunk, lines)
  end
  return lines
end


local function loadFile(path) 
  local f = io.input(path)
  local data = {}
  while loadChunk(f, data) do end
  return parseLines(data)
end  

local function splitLines(lineTable)
  for i=1, #lineTable do
    lineTable[i] = string.split(lineTable[i], ",")
  end
end

function create_events(line)
  redis_client:del(opt.redis_prefix .. "-events")

  local events = string.split(line, ",")
  for i=1, #events do
    redis_client:sadd(opt.redis_prefix .. "-events", events[i])
  end
  return #events
end

local function update_data_split(num_obs)
  data_split["training"]["start"] = math.ceil(2)
  data_split["training"]["stop"] = math.floor(num_obs * 0.5)
  data_split["validation"]["start"] = math.floor(num_obs * 0.5) + 1
  data_split["validation"]["stop"] = math.floor(num_obs * 0.75)
  data_split["testing"]["start"] = math.floor(num_obs * 0.75) + 1
  data_split["testing"]["stop"] = math.floor(num_obs * 1.0)
end

function loadData()
  raw_data = loadFile(opt.data)
  num_var = create_events(raw_data[1])
 
  for i=2, #raw_data do
    raw_data[i] = string.split(raw_data[i], ",")
    raw_data[i][1] = timestamp2theta(raw_data[i][1])
    for j=2, #raw_data[i] do  
      raw_data[i][j] = tonumber(raw_data[i][j])
    end
  end  
  
  print ("normalizing data")
  local tmp = torch.DoubleTensor(#raw_data-1)
  for i = 1, num_var-1 do
    xlua.progress(i, num_var)

    tmp:zero()
    for j=2, #raw_data do
      tmp[j-1] = raw_data[j][i+1]
    end
    local mean = torch.mean(tmp)
    local std = torch.std(tmp)
    for j=2, #raw_data do
      raw_data[j][i+1] = (raw_data[j][i+1] - mean)/std
    end

  end
  
  
  
  --[[
  for i=1, num_events do
    distributions:sub(1,-1,i,i):add(-torch.mean(distributions:sub(1,-1,i,i)))
  end
  distributions:div(torch.std(distributions))
  
  print ("distributions", distributions)
  ]]--
  
  update_data_split(#raw_data -1)
  print ("data_split", data_split)
  
end


function loadBatch(size, len_seq)
  local batch = {}
  
  while #batch < size do 
    local distributions = torch.DoubleTensor(len_seq+1, num_var)
    local timestamps = torch.DoubleTensor(len_seq+1, theta_size)
    local i_start = torch.random(data_split[batch_type]["start"], data_split[batch_type]["stop"]-opt.len_seq)
    
    for i=i_start, i_start+len_seq do 
      --local line =  string.split(raw_data[i], ",")
      timestamps[i-i_start + 1] = raw_data[i][1]
      for j=2, #raw_data[i] do
        distributions[i-i_start+1][j-1] = raw_data[i][j]
      end
    end
    local seq = {["timestamps"] = timestamps,
                 ["distributions"] = distributions
                }
    
    table.insert(batch, seq)
    
  end
    
  collectgarbage()
  return batch
end


function nextBatch(size, len_seq)
  local batch = {}
  local sources = {}

  while #batch < size and #testing_data > 0 do 
    local sample = table.remove(testing_data)
    local seq = loadSeq(sample)
    if #seq > math.ceil(len_seq) then
      seq = splitLines2(seq, len_seq)
      table.insert(sources, sample[1])
      table.insert(batch, seq)
    end
  end
  collectgarbage()
  return batch, sources
end

function donkey_init()
  loadData()
end

