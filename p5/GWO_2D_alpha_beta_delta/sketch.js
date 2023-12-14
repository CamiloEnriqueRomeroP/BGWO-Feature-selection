var C_x, a_x, A_x
var C_y, a_y, A_y
var wolf_img, bofallo_img;
var tickbox

var alphaWolf = {
  x: 400,
  y: 400,
  D: {
    x: 0,
    y: 0
  }
};

var betaWolf = {
  x: 500,
  y: 700,
  D: {
    x: 0,
    y: 0
  }
};

var deltaWolf = {
  x: 200,
  y: 600,
  D: {
    x: 0,
    y: 0
  }
};

var omegaWolf = {
  x_t1: 400,
  y_t1: 600,
  x_t2: 0,
  y_t2: 0
};


function setup() {
  createCanvas(1050, 800);
  background(246);
  

  C_x = createSlider(0, 2, 1, 0.1)
  C_y = createSlider(0, 2, 1, 0.1)
  a_x = createSlider(0, 2, 1, 0.1)
  a_y = createSlider(0, 2, 1, 0.1)
  r1_x = createSlider(0, 1, 0.75, 0.01)
  r1_y = createSlider(0, 1, 0.75, 0.01)
  r2_x = createSlider(0, 1, 0.5, 0.01)
  r2_y = createSlider(0, 1, 0.5, 0.01)
  A_x = createSlider(-5, 5, 1, 0.1)
  A_y = createSlider(-5, 5, 1, 0.1)
  tickbox = createCheckbox('Tick for random r1 and r2', false);
  



  a_x.position(50, 50)
  a_y.position(50, 75)
  r1_x.position(50, 100)
  r1_y.position(50, 125)
  r2_x.position(50, 150)
  r2_y.position(50, 175)
  A_x.position(50, 200 + 20)
  A_y.position(50, 225 + 20)
  C_x.position(50, 250 + 20)
  C_y.position(50, 275 + 20)
  
  tickbox.position(50,330);


  wolf_img = loadImage('wolf.png')
  bofallo_img = loadImage('bofallo.png')
}

function draw() {
  background(246);
  rect(0,0,width,height)
  
  fill(0,0,0)
 
  text("Click and drag wolves and the circles", 450,50)

  textSize(20)
  fill(0)

  text("a_x = " + a_x.value(), 250, 70)
  text("a_y = " + a_y.value(), 250, 90)

  text("r1_x = " + r1_x.value(), 250, 120)
  text("r1_y = " + r1_y.value(), 250, 140)

  text("r2_x = " + r2_x.value(), 250, 170)
  text("r2_y = " + r2_y.value(), 250, 190)

  text("A_x = " + A_x.value(), 250, 220 + 20)
  text("A_y = " + A_y.value(), 250, 240 + 20)

  text("C_x = " + C_x.value(), 250, 270 + 20)
  text("C_y = " + C_y.value(), 250, 290 + 20)



  A_x.value(2 * a_x.value() * r1_x.value() - a_x.value());
  A_y.value(2 * a_y.value() * r1_y.value() - a_y.value());

  C_x.value(2 * r2_x.value());
  C_y.value(2 * r2_y.value());

  fill(255, 0, 0)
  // circle(p_Slider_x.value(), p_Slider_y.value() , 50)





  noFill()


  alphaWolf.D.x = abs(C_x.value() * alphaWolf.x - omegaWolf.x_t1)
  alphaWolf.D.y = abs(C_y.value() * alphaWolf.y - omegaWolf.y_t1)


  betaWolf.D.x = abs(C_x.value() * betaWolf.x - omegaWolf.x_t1)
  betaWolf.D.y = abs(C_y.value() * betaWolf.y - omegaWolf.y_t1)


  deltaWolf.D.x = abs(C_x.value() * deltaWolf.x - omegaWolf.x_t1)
  deltaWolf.D.y = abs(C_y.value() * deltaWolf.y - omegaWolf.y_t1)


  X1_alpha = alphaWolf.x - A_x.value() * alphaWolf.D.x;
  Y1_alpha = alphaWolf.y - A_y.value() * alphaWolf.D.y;

  X1_beta = betaWolf.x - A_x.value() * betaWolf.D.x;
  Y1_beta = betaWolf.y - A_y.value() * betaWolf.D.y;

  X1_delta = deltaWolf.x - A_x.value() * deltaWolf.D.x;
  Y1_delta = deltaWolf.y - A_y.value() * deltaWolf.D.y;

  fill(0)


  omegaWolf.x_t2 = (X1_alpha + X1_beta + X1_delta) / 3;
  omegaWolf.y_t2 = (Y1_alpha + Y1_beta + Y1_delta) / 3;


  line(omegaWolf.x_t1, omegaWolf.y_t1, omegaWolf.x_t2, omegaWolf.y_t2)

  
  
  image(wolf_img, alphaWolf.x, alphaWolf.y, 70, 70)
  text('Alpha', alphaWolf.x, alphaWolf.y)
  
  image(wolf_img, betaWolf.x, betaWolf.y, 70, 70)
  text('Beta', betaWolf.x, betaWolf.y)
 
  image(wolf_img, deltaWolf.x, deltaWolf.y, 70, 70)
  text('Delta', deltaWolf.x, deltaWolf.y)
  
  image(wolf_img, omegaWolf.x_t2 - 40, omegaWolf.y_t2 - 40, 70, 70)
  text('Omega', omegaWolf.x_t2-60, omegaWolf.y_t2-40)
  
  noFill()
  circle(omegaWolf.x_t1, omegaWolf.y_t1, 50)


  noFill()

  rect(40, 40, 320, 170)

  if (tickbox.checked()){
    r1_x.value(random(0,1))
    r1_y.value(random(0,1))
    r2_x.value(random(0,1))
    r2_y.value(random(0,1))
  }

  handleClicks()

}

function handleClicks() {

  distance2alpha = dist(mouseX, mouseY, alphaWolf.x+35, alphaWolf.y+35)
  distance2beta = dist(mouseX, mouseY, betaWolf.x+35, betaWolf.y+35)
  distance2delta = dist(mouseX, mouseY, deltaWolf.x+35, deltaWolf.y+35)
  
  distance2omegat1 = dist(mouseX, mouseY, omegaWolf.x_t1, omegaWolf.y_t1)

  threshold = 40;



  if (mouseIsPressed == true) {

    if (distance2alpha < threshold) {
      alphaWolf.x = mouseX-35;
      alphaWolf.y = mouseY-35;
    }

    if (distance2beta < threshold) {
      betaWolf.x = mouseX-35;
      betaWolf.y = mouseY-35;
    }

    if (distance2delta < threshold) {
      deltaWolf.x = mouseX-35;
      deltaWolf.y = mouseY-35;
    }
    
    if (distance2omegat1 < threshold){
      omegaWolf.x_t1 = mouseX;
      omegaWolf.y_t1 = mouseY;
    }
    
    
  }

}