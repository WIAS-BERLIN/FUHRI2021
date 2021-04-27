
using ImageView
using Images
using Gtk
using ForwardDiff
using DiffResults


#####
#####
#
# PROJECT DESCRIPTION AND TASKS
#
####
####




## Newton method from previous exercises (see Pluto notebook)
function newton_advanced(F::Function, init; maxits = 10, tol = 1e-12)
    zn = init
    result = DiffResults.JacobianResult(zn, zn)
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
            return zn, DiffResults.value(result), it, false
        else
            zn = zn - DiffResults.gradient(result) \ DiffResults.value(result)
        end
    end
end


## FULL CONVOLUTION FILTER
function apply_convolution!(result, input, filter)
    input_r, input_c = size(input)
    filter_r = convert(Int32,sqrt(length(filter)-1))
    result_r, result_c = size(result)
    filter_middle = convert(Int32, (filter_r + 1)//2)
    m = convert(Int32, (filter_r - 1)//2)
    shifts::Array{Int32,1} = -m:m
    II::Array{Int32,1} = zeros(Int32,length(shifts)^2)
    JJ::Array{Int32,1} = zeros(Int32,length(shifts)^2)
    for k = 1 : length(shifts)
        for i = 1 : length(shifts)
            II[(k-1)*length(shifts)+i] = shifts[k]
            JJ[(i-1)*length(shifts)+k] = shifts[k]
        end
    end
    Ii::Int = 0
    Ij::Int = 0
    fill!(result,0.0)
    for i in 1:result_r
        for j in 1:result_c
            for f = 1 : length(filter) - 1
                Ii = max(min(i+II[f],input_r),1)
                Ij = max(min(j+JJ[f],input_c),1)
                result[i,j] += input[Ii,Ij] * filter[f]
            end
            result[i,j] += filter[end]
        end
    end
end


## COMPUTE DIFERENCE OF TWO IMAGES (IN THEIR GRAY-VALUES)
function graydiff(img1::Array{T1,2},img2::Array{T2,2}) where {T1,T2}
    s::T2 = 0.0
    for i = 1 : size(img1,1), j = 1 : size(img1,2)
        s += (img1[i,j] - img2[i,j])^2
    end
    return s
end


function main()
    # load source and target image
    source_img = load("res/juergen_256.jpg")
    target_img = load("res/juergent_256.png")
    source_gray = convert(Array{Float64,2},Gray.(source_img))
    target_gray = convert(Array{Float64,2},Gray.(target_img))

    # choose filters for approximating the target image
    n=3 # filter-size
    filter = rand(n^2)/n^2; append!(filter,[0]) # init value
    filter_function = apply_convolution!

    # setup objective function
    img_result = Array{Float64,2}(deepcopy(source_gray))
    function objective_function(filter)
        if eltype(filter) != eltype(img_result)
            # if DualNumbers appear the array that stores the resulting image has to change its type
            img_result = zeros(eltype(filter),size(target_gray))
        end
        filter_function(img_result,source_gray, filter)
        return graydiff(target_gray,img_result)
    end
    

    ################################
    ### INSERT OPTIMISATION HERE ###
    ################################


    # show resulting filter and apply it to image
    @show filter
    img_result = Array{Float64,2}(deepcopy(source_gray))
    filter_function(img_result,source_gray, filter)
    target_gray = Array{Float64,2}(target_gray)

    # print distance between the resulting image and the target image
    println("| bestapprox - target | = $(graydiff(target_gray,img_result))")

    # plot all images
    @info "Plotting results: source | best-with-filter | target"
    gui = imshow_gui((800, 200), (1, 3)) 
    canvases = gui["canvas"]
    imshow(canvases[1,1], source_gray)
    imshow(canvases[1,2], img_result)
    imshow(canvases[1,3], Array{Float64,2}(target_gray))
    Gtk.showall(gui["window"])
end

main()