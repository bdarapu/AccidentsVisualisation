function ScatterPlot(filename) {
    
    if(svg!=undefined)
        svg.selectAll("*").remove();
    var xScale = d3.scaleLinear().range([left_pad, w-pad]);

    var xAxis = d3.axisBottom(xScale);
    
    var yScale = d3.scaleLinear().range([pad, h-pad*2]);
    var yAxis = d3.axisLeft(yScale);
    
    
    color = "008C00";
    
    d3.csv(filename, function(d) {
            
            d.a1 = +d.a1;
            d.a2 = +d.a2;
            
            return d;
        },function(error,data){
            if(error) throw error;
        

        var xValueR = function(d) { return d.a1;};
        var yValueR = function(d) { return d.a2;};
        
        
        xScale.domain([d3.min(data, xValueR), d3.max(data, xValueR)]);
        
        yScale.domain([d3.min(data, yValueR), d3.max(data, yValueR)]);
        
        
        svg.append("g")
          .attr("class", "axis")
          .attr("transform", "translate(0, "+(h-pad)+")")
          .call(xAxis);
          
            
 
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
        .text("Component1");

        svg.append("text")
        
        .attr("y", left_pad+210)
        .attr("x",h+225)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Component2");


        svg.selectAll(".circle")
            .data(data)
            .enter()
            .append("circle")
            .attr("r", 2.5)
            .attr("cx", function(d){
               
                return xScale(d.a1);
            }) 
            .attr("cy", function(d){
                return yScale(d.a2);
            }) 
            
            .style("fill","blue")

            .attr("stroke", "black")
            
            ;
    });

}