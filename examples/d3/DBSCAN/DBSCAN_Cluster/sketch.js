let dbscanModel;
const width = 640;
const height = 480;
const colDict = {
  0: "skyblue",
  1: "coral",
  2: "olive",
  3: "tan",
  4: "grey",
};

//  ----- Initialize the example: ------ //
function init() {
  // make all those nice buttons
  createButtons();
  // start with 3 cluster
  make(1.55);
}
init();

// STEP 1:
// create all the buttons
function createButtons() {
  addClusterButton(1.55);
  addClusterButton(1.56);
  addClusterButton(2);
  addClusterButton("circle");
  addClusterButton("moon");
}

// STEP 2:
// create the model
function make(eps) {
  const options = {
    eps: eps,
    minPts: 3,
  };
  // if moon or circle data, set the options to 0.1 and 0.16 eps
  if (eps === "moon") {
    options.eps = 0.1;
  } else if (eps === "circle") {
    options.eps = 0.16;
  }
  console.log(eps, options.eps);

  // get the path to the data in our data folder dynamically
  const dataPath = `data/gaussian2d_${eps}clusters.csv`;
  // create a new dbscan clustering each time on make()
  dbscanModel = ml5.dbscan(dataPath, options, modelReady);
}

// Step 3:
// when the model is ready, make the chart
function modelReady() {
  console.log(dbscanModel.dataset);
  makeChart();
}

// Step 4:
// use the fancy d3 to make magic
function makeChart() {
  const { dataset } = dbscanModel;
  // clear the chart each time
  // less efficient, but simple
  d3.select("svg").remove();

  // reappend the svg to the chart area
  const svg = d3
    .select("#chart")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  // d[0] is for the x value in the array
  const xScale = d3
    .scaleLinear()
    .domain(d3.extent(dataset, d => d[0]))
    .range([10, width - 100]);

  // d[1] is for the y value in the array
  const yScale = d3
    .scaleLinear()
    .domain(d3.extent(dataset, d => d[1]))
    .range([height - 50, 20]);

  svg
    .selectAll("circle")
    .data(dataset)
    .enter()
    .append("circle")
    .attr("cx", d => xScale(d[0]))
    .attr("cy", d => yScale(d[1]))
    .attr("r", 6)
    .attr("fill", "black");

  d3.select("svg")
    .selectAll("circle")
    .transition()
    .attr("fill", (d, i) => colDict[dataset[i].clusterid]);
}

// adds the buttons for the respective cluster data
// we could also use d3.append() and d3.select() here :)
function addClusterButton(eps, minPts = 3) {
  const btn = document.createElement("BUTTON");
  btn.innerText = `cluster: ${eps} & minPts: ${minPts}`;

  btn.addEventListener("click", function(e) {
    make(eps);
  });

  document.querySelector("#buttons").appendChild(btn);

  return btn;
}
