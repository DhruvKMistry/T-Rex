var actionInterval = 160;

var numGames = 1;
var games  = new Array();

for (var i = 0; i < numGames; ++i) {
  var dinoName = ".dino" + i;
  games.push(new Runner(dinoName));
}

var player = new Player();
var human = true;

function grey(input) {
    cnx.drawImage(myimage, 0 , 0);
    var width = input.width;
    var height = input.height;
    var imgPixels = cnx.getImageData(0, 0, width, height);
    
    for(var y = 0; y < height; y++){
        for(var x = 0; x < width; x++){
            var i = (y * 4) * width + x * 4;
            var avg = (imgPixels.data[i] + imgPixels.data[i + 1] + imgPixels.data[i + 2]) / 3;
            imgPixels.data[i] = avg;
            imgPixels.data[i + 1] = avg;
            imgPixels.data[i + 2] = avg;
        }
    }
    
    cnx.putImageData(imgPixels, 0, 0, 0, 0, imgPixels.width, imgPixels.height);
}

// Deep Q Learning parameters
var num_inputs =881; // X,Y,grayscale, num_inputs * temporal_window + num_actions * temporal_window + num_inputs
var num_actions = 2; // JUMP, IDLE, DUCK
var temporal_window = 1;

var layer_defs = [];
layer_defs.push({type:'input', out_sx:42, out_sy:42, out_depth:1}); // declare size of input
// output Vol is of size 32x32x3 here
layer_defs.push({type:'conv', sx:4, filters:4, stride:1, pad:1, activation:'relu'});
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Vol of the same size
// output Vol will thus be 32x32x16 at this point
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'conv', sx:2, filters:8, stride:1, pad:1, activation:'relu'});
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Vol of the same size
// output Vol will thus be 32x32x16 at this point
layer_defs.push({type:'pool', sx:2, stride:2});
layer_defs.push({type:'regression', num_neurons:num_actions});

var tdtrainer_options = {learning_rate:0.01, momentum:0.9, batch_size:1024, l2_decay:0.01};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 30000;
opt.start_learn_threshold = 1000;
//opt.start_learn_threshold = 70;
opt.gamma = 0.7;
opt.learning_steps_total = 200000;
opt.learning_steps_burnin = 3000;
opt.epsilon_min = 0.05;
opt.epsilon_test_time = 0.05;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;
opt.random_action_distribution = [0.1, 0.9];

var brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

if (sessionStorage.getItem("model") != null)
{
    console.log('Play by AI');
    console.log(sessionStorage.getItem("model"));
    var j = JSON.parse(sessionStorage.getItem("model"));
    brain.value_net.fromJSON(j);
    human=false;
    sessionStorage.removeItem("model");
}
else
{
    console.log('Play by human');
    human=true;
}

function getObstacleType(obst) {
  if (obst.typeConfig.type == "CACTUS_LARGE") {
    return 0;
  } else if (obst.typeConfig.type == "CACTUS_SMALL") {
    return 1;
  } else if (obst.typeConfig.type == "PTERODACTYL") {
    return 2;
  }
}

function makeStep(idx) {
  var obstacleDetected = false;

  if (!games[idx].started) { // start
    games[idx].playIntro();
    games[idx].play();
  } else if (games[idx].activated) { // playing
    var currentSpeed = games[idx].currentSpeed;

    // no obstacles
    if (games[idx].horizon.obstacles.length == 0) {
      obstacleDetected = false;
    }
    else {
        
      var obst = games[idx].horizon.obstacles[0];
      var tRex_xPos = games[idx].tRex.xPos;
        
      if (tRex_xPos < obst.xPos) {
        obstacleDetected = true;
      }
    }
      var base64 = document.getElementsByClassName('runner-canvas')[0].toDataURL().substring(22);
      //console.log(base64);
      imgElem.setAttribute('src', "data:image/jpg;base64," + base64);
      
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      imgElem.onload = function(){
          ctx.drawImage(imgElem, 0, 0, 300, 500, 0, 0, 42, 42); // crop , resize
      }

      var image = new Image();
      image.src = canvas.toDataURL();
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
    // train by image to get action
      var action = brain.forward(convnetjs.img_to_vol(image));
      
      if(!human)
      {
        if (action == 0)
          player.do(Player.actions.IDLE);
        else if (action == 1)
          player.do(Player.actions.JUMP);
      }

    var nothing = true;

    if (obstacleDetected) {
      var obst0 = games[idx].horizon.obstacles[0];
      var tRex_xPos = games[idx].tRex.xPos;

      if ((tRex_xPos+50) >= (obst0.xPos + obst0.width)) {
        brain.backward(10.0);
        console.log(idx);
      }
      else {
        nothing = false;
      }
    }

    if (nothing) {
      brain.backward(0.0);
    }
  }
  // finish
  else {
    brain.backward(-1.0);
    games[idx].restart();
  }
}

setInterval(function() {
  makeStep(0);
}, actionInterval);
