$(document).on('change', '#image-selector', function() {

    let reader = new FileReader();
    reader.onload= function(){
        let dataUrl = reader.result;
        $('#selected-image').attr('src',dataUrl);
        $('#predict').empty();
       
    }
    let file = $('#image-selector').prop('files')[0];
    reader.readAsDataURL(file)
});
let model;
(async function(){
    model= await tf.loadLayersModel('http://localhost:81/model/model.json');
    $('#pro').hide()

})();
$(document).on('click', '#predict-button', async function() { 
    let image = $('#selected-image').get(0);
    
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([224,224]).toFloat().expandDims();
    let prediction = await model.predict(tensor).data();
    let top3 = Array.from(prediction)
    .map(function(p,i){
        return {
            probab: p,
            classname:CANCER_CLASSES[i]
        };

    }).sort(function(a,b){
        return b.probab-a.probab;
    }).slice(0,4);
    $("#predict").empty();
    top3.forEach(function(p){
        $('#predict').append(`<li>${p.classname}: ${p.probab}</li>`);
    });
});