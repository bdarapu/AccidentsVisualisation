function ScatterPlot(filename) {
    
    if(svg!=undefined)
        svg.selectAll("*").remove();
    var xScale = d3.scaleLinear().range([left_pad, w-pad]);

    var xAxis = d3.axisBottom(xScale);
    
    var yScale = d3.scaleLinear().range([pad, h-pad*2]);
    var yAxis = d3.axisLeft(yScale);
    
    // color = ["008C00","C3D9FF"];
    color = "008C00";
    // Load data
    d3.csv(filename, function(d) {
            // console.log(filename)
            d.a1 = +d.a1;
            d.a2 = +d.a2;
            // d.type = +d.type;
            // console.log(d);
            return d;
        },function(error,data){
            if(error) throw error;
        // });

        var xValueR = function(d) { return d.a1;};
        var yValueR = function(d) { return d.a2;};
        /*var xScale = d3.scaleLinear().range([0, width]);
        var yScale = d3.scaleLinear().range([height, 0]);*/
        
        xScale.domain([d3.min(data, xValueR), d3.max(data, xValueR)]);
        // console.log(xScale);
        yScale.domain([d3.min(data, yValueR), d3.max(data, yValueR)]);
        
        
        svg.append("g")
          .attr("class", "axis")
          .attr("transform", "translate(0, "+(h-pad)+")")
          .call(xAxis);
          // .append("text")
          // .attr("x", (w-pad - 50))
          //   // .attr("y", (h-pad))
          //   // .attr("dy", ".5em")
          //   .style("text-anchor", "middle")
          //   .text("Component B")
            
 
        svg.append("g")
          .attr("class", "axis")
          .attr("transform", "translate("+(left_pad-pad)+", 0)")
          .call(yAxis);

        svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", left_pad-80)
        .attr("x",h-400)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("PCA Component1");

        svg.append("text")
        //.attr("transform", "rotate(-20)")
        .attr("y", left_pad+210)
        .attr("x",h+225)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("PCA Component2");


        svg.selectAll(".circle")
            .data(data)
            .enter()
            .append("circle")
            .attr("r", 2.5)
            .attr("cx", function(d){
                // console.log(xScale(d.a1));
                return xScale(d.a1);
            }) 
            .attr("cy", function(d){
                return yScale(d.a2);
            }) 
            /*.style("fill", function(d) {
                return color[d.type-1];
            })*/
            .style("fill","blue")

            .attr("stroke", "black")
            //.attr("stroke-width", function(d) {return d/2;});
            ;
    });

}