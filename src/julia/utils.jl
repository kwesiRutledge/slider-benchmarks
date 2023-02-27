"""
utils.jl
Description

"""

"""
circleShape(h, k, r)
Description:
    This defines a circle where (h,k) is the center and radius is r.
"""
function circleShape(h, k, r)
    theta = LinRange(0,2*pi,500)
    h .+ r*sin.(theta), k .+ r*cos.(theta)
end

