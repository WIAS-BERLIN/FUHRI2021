#
using ImageView
using Images
using ForwardDiff
using DiffResults


## Newton method from previous exercises (see Pluto notebook)
function newton_advanced(F::Function, init; maxits = 10, tol = 1e-12)
    zn = init
    result = DiffResults.JacobianResult(F(zn), zn)
    cfg = ForwardDiff.JacobianConfig(F,zn)
    it::Int = 0
    while (true)
        it += 1
        @info "Iteration $it..."
        result = ForwardDiff.jacobian!(result,F,zn,cfg)
        if sqrt(sum(DiffResults.value(result).^2)) < tol
            return zn, DiffResults.value(result), it, true
        elseif it >= maxits
            return zn, DiffResults.value(result), it, false
        elseif any(isnan, zn)
            # Halt due to NaN
            return zn, Diffresults.value(result), it, false
        else
            zn = zn - DiffResults.gradient(result) \ DiffResults.value(result)
        end
    end
end

# load source and target image
source_img = load("res/juergen_hybrid_256.jpg")
target_img = load("res/juergen_256.jpg")
target_gray = Array{Float64,2}(Gray.(target_img))
source_gray = convert(Array{Float64,2},Gray.(source_img))

# maximal allowed convolution filter size
n=5

## function that applies the filter convolution to the input image and writes it to result
function apply_convolution!(result, input, filter)
    input_r, input_c = size(input)
    filter_r = convert(Int32,sqrt(length(filter)))
    result_r, result_c = size(result)
    filter_middle = convert(Int32, (filter_r + 1)//2)
    m = convert(Int32, (filter_r - 1)//2)
    Ii::Int = 0
    Ij::Int = 0
    fill!(result,0)
    for i in 1:result_r
        for j in 1:result_c
            for fi in -m:m
                for fj in -m:m
                    Ii = mod(i + fi - 1,input_r) + 1
                    Ij = mod(j + fj - 1,input_c) + 1
                    result[i,j] += input[Ii,Ij]*filter[(filter_middle+fi-1)*filter_r+filter_middle+fj]
                    # extend for an arbitrary filter size
                end
            end
        end
    end

    result = min.(result, 1.0)
    result = max.(result, 0.0)
end

# computes gray difference of two images given as Arrays
function graydiff(img1,img2)
    return sum((view(img1,:,:) .- view(img2,:,:)).^2)
end

## objective function to be minimised
function objective_function(filter)
    img_result = Array{eltype(filter),2}(deepcopy(source_gray))
    apply_convolution!(img_result,source_gray, filter)
    return graydiff(target_gray,img_result) + sum(filter.^2)
end

# initial filter = n xn Matrix written as a vector of length n^2
filter = rand(n^2)/n^2 #3.3e-1*[1 1 1; 0 1 1; 1 0 0]

# run Newton on gradient of objective_function
@info "Starting optimisation..."
dobj(filter) = ForwardDiff.gradient(objective_function,filter)
filter, df, its = newton_advanced(dobj, filter; maxits = 5, tol = 1e-10)

# show result and apply filter
@show filter
img_result = Array{Float64,2}(deepcopy(source_gray))
apply_convolution!(img_result,source_gray, filter)
@show graydiff(target_gray,img_result)

# plot
@info "Plotting results: source | best-with-filter | target"
gui = imshow_gui((900, 300), (1, 3)) 
canvases = gui["canvas"]
imshow(canvases[1,1], source_gray)
imshow(canvases[1,2], img_result)
imshow(canvases[1,3], target_gray)
Gtk.showall(gui["window"])