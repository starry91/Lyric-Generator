$(document).ready(function () {
  $("#submit_text").click(function () {
    var search_text = $("#search_text").val();
    console.log(search_text);
    $("#results").html('');
    $("#optiondiv").html('');
    $.ajax({
        url: `/lyrics`,
        contentType: "application/json",
        type: 'POST',
        dataType: 'json',
        data: JSON.stringify({
          search_text: search_text
        }),
        success: function (data) {
          $("#results").html('');
          console.log(data);
          $("#results").append(data.generated);
          // var valid_search = data.valid_search;
          // if (valid_search==true){
          //   var event_df = data.event_df;
          //   $("#results").append(event_df);
          // }else {
          //   var search_result = data.search_result;
          //   console.log('the search_results', search_result);
          //
          //   var searches = "<select id = 'option_texts'>  <option>Select an option"
          //   for (var i = 0; i < Math.min(search_result.length, 10); i++) {
          //     // searches = searches + "<a href = 'https://google.com' >" + search_result[i] + "</a><br>";
          //     searches = searches + "  <option value='" + search_result[i] + "'>" + search_result[i]+ "</option>'";
          //   }
          //   searches = searches + "</select>";
          //   console.log(searches);
          //   $("#optiondiv").append("<br><p align = 'left'><b>There are multiple wikipedia articles with that word. Please select an option from the dropdown.</b></p><br>" + searches);
          //   $("#option_texts").change(function(){
          //     console.log("The text has been changed.");
          //     var search_text = $("#option_texts").val();
          //     console.log(search_text);
          //     $.ajax({
          //         url: `/search`,
          //         contentType: "application/json",
          //         type: 'POST',
          //         dataType: 'json',
          //         data: JSON.stringify({
          //           search_text: search_text
          //         }),
          //         success: function (data) {
          //           $("#results").html('');
          //           console.log(data);
          //             var event_df = data.event_df;
          //             $("#results").append(event_df);
          //           }
          //         })
          //
          //
          //   });
          //
          //
          //
          // }
          }
          })
  })
})