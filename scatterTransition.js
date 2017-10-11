
function plot_values(da, type,side, data_type)
{
   
      d3.selectAll("svg").remove();

if(type == 'pca' && data_type == 'random')
{
  data = da.pca_random;
  
}
else if(type == 'pca' && data_type == 'stratified')
{
  data = da.pca_stratified;
  
}

else if(type == 'correlation' && data_type == 'random')
  data = da.random_corr_vals;
else if(type == 'correlation' && data_type == 'stratified')
  data = da.stratified_corr_vals;

else if(type == 'euclidean' && data_type == 'random')
  data = da.random_euclidean_vals;
else if(type == 'euclidean' && data_type == 'stratified')
  data = da.stratified_euclidean_vals;


console.log(data);
  id = '#scatter_plot_'+side;
  console.log(id);


    var margin = {top: 20, right: 15, bottom: 60, left: 60}
      , width = 960 - margin.left - margin.right
      , height = 500 - margin.top - margin.bottom;

    var x = d3.scale.linear()
//              .domain([0, d3.max(data, function(d) { return d[0]; })])
              .domain([d3.min(data, function(d) { return d[0]; }), d3.max(data, function(d) { return d[0]; })])
              .range([ 0, width ]);

    var y = d3.scale.linear()
//            .domain([0, d3.max(data, function(d) { return d[1]; })])
              .domain([d3.min(data, function(d) { return d[1]; }), d3.max(data, function(d) { return d[1]; })])
              .range([ height, 0 ]);

    var chart = d3.select(id)
    .append('svg:svg')
    .attr('width', width + margin.right + margin.left)
    .attr('height', height + margin.top + margin.bottom+25)
    .attr('class', 'chart');

    var main = chart.append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
    .attr('width', width)
    .attr('height', height)
    .attr('class', 'main')

    // draw the x axis
    var xAxis = d3.svg.axis()
    .scale(x)
    .orient('bottom');

    main.append('g')
    .attr('transform', 'translate(0,' + height + ')')
    .attr('class', 'main axis date')
    .call(xAxis)
  .append("text")
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", "-.55em")
      .attr("transform", "translate(900,0)" )
      .text("PCA1")
      .style('font','20px times');

    // draw the y axis
    var yAxis = d3.svg.axis()
    .scale(y)
    .orient('left');

    main.append('g')
    .attr('transform', 'translate(0,0)')
    .attr('class', 'main axis date')
    .call(yAxis)
  .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -55)
      .attr("dy", ".80em")
      .style("text-anchor", "end")
      .text("PCA2")
      .style('font','20px times');

    var g = main.append("svg:g");

    g.selectAll("scatter-dots")
      .data(data)
      .enter().append("svg:circle")
          .attr("cx", function (d,i) {
           return x(0);
         } )
          .attr("cy", function (d) { 
          return y(0);
          } )
          .attr("r", 4)
          .attr("fill", "steelblue")
                  .transition()
        .duration(2000)
          .attr("cx", function (d,i) { return x(d[0]); } )
        .attr("cy", function(d) {
            return y(d[1]);
          });
}