# Raissi et al plotting scripts - https://github.com/maziarraissi/PINNs/blob/master/Utilities/plotting.py
# All code in this script is credited to Raissi et al


import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


import matplotlib.pyplot as plt

def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def plot_solution_domain1D(model, domain, ub, lb, Exact_u=None, u_transpose=False):
    """
    Plot a 1D solution Domain
    Arguments
    ---------
    model : model
        a `model` class which contains the PDE solution
    domain : Domain
        a `Domain` object containing the x,t pairs
    ub: list
        a list of floats containing the upper boundaries of the plot
    lb : list
        a list of floats containing the lower boundaries of the plot
    Exact_u : list
        a list of the exact values of the solution for comparison
    u_transpose : Boolean
        a `bool` describing whether or not to transpose the solution plot of the domain
    Returns
    -------
    None
    """
    X, T = np.meshgrid(domain[0],domain[1])

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    if Exact_u is not None:
        u_star = Exact_u.T.flatten()[:,None]

    u_pred, f_u_pred = model.predict(X_star)
    if u_transpose:
        U_pred = griddata(X_star, u_pred.T.flatten(), (X, T), method='cubic')
    else:
        U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    fig, ax = newfig(1.3, 1.0)

    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    len_ = len(domain[1])//4

    line = np.linspace(domain[0].min(), domain[0].max(), 2)[:,None]
    ax.plot(domain[1][len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][2*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(domain[1][3*len_]*np.ones((2,1)), line, 'k--', linewidth = 1)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    leg = ax.legend(frameon=False, loc = 'best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('u(t,x)', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/2, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    Zoom_limit = 0.04 + Exact_u[:,:].max().item()   #To see the curve better. Smaller number, Bigger zoom


    ax = plt.subplot(gs1[0, 0])
    ax.plot(domain[0],Exact_u[:,len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title('t = %.2f' % (domain[1][len_]), fontsize = 10)
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim([-Zoom_limit, Zoom_limit])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(domain[0],Exact_u[:,2*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[2*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim([-Zoom_limit, Zoom_limit])
    ax.set_title('t = %.2f' % (domain[1][2*len_]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(domain[0],Exact_u[:,3*len_], 'b-', linewidth = 2, label = 'Exact')
    ax.plot(domain[0],U_pred[3*len_,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim([-Zoom_limit, Zoom_limit])
    ax.set_title('t = %.2f' % (domain[1][3*len_]), fontsize = 10)

    plt.show()


def plot_solution_domain2D(model, domain, ub, lb, Exact_u=None, u_transpose=False):
    """
    Plot a 2D solution Domain
    Arguments
    ---------
    model : model
        a `model` class which contains the PDE solution
    domain : Domain
        a `Domain` object containing the x,y,t pairs
    ub: list
        a list of floats containing the upper boundaries of the plot
    lb : list
        a list of floats containing the lower boundaries of the plot
    Exact_u : list
        a list of the exact values of the solution for comparison
    u_transpose : Boolean
        a `bool` describing whether or not to transpose the solution plot of the domain
    Returns
    -------
    None
    """
    X, Y, T = np.meshgrid(domain[0],domain[1], domain[2])

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))
    if Exact_u is not None:
        u_star = Exact_u.T.flatten()[:,None]

    u_pred, f_u_pred = model.predict(X_star)
    
    #2D slices through the predicted surface for comparison
    mesh = 100
    t = domain[2]
    t_slices = 5
    t_delta = ub[2]/t_slices           #time jump to get snapshot
    print("Time step: ", t_delta)
    ######################################Prepare mesh for comparison#######################################
    xm, ym = np.meshgrid(domain[0],domain[1])
    #######################################Zoom Factor######################################################
    Zoom_limit = 0.001 + Exact_u[:,:,:].max().item()   #To see the curve better. Smaller number, Bigger zoom
    u_pred_reshaped = u_pred.reshape(100,100,100)
    ################################################Slices#################################################
    for t_slice in range(t_slices+1):
        fig = plt.figure(figsize=(600, 250), dpi=80)
        fig.subplots_adjust(left=0.1, bottom=0.5, right=0.125, top=0.6,
                    wspace=0.3, hspace=None)
        print("Slice #", t_slice)
        if t_slice == t_slices:
            at_index = mesh - 1
        else:
            at_index = int(t_slice * mesh/t_slices)
        print("slicing at index ", at_index)
    
        #############################################################################
        ########################Lay planes on each other#############################
        
        ax = fig.add_subplot(t_slices,3, 1, projection='3d')
        surf = ax.plot_surface(xm,ym, Exact_u[:,:, at_index].T, cmap=mpl.cm.gist_heat, label = 'Exact')
        ax.plot_wireframe(xm,ym,  u_pred_reshaped[:,:, at_index], rstride=2, cstride=2,label = 'Prediction')
        ax.view_init(45, 45)
        ax.set_xlabel('x',fontsize = 15)
        ax.set_ylabel('y',fontsize = 15)
        ax.set_zlabel('u(x,y,t)')
        ax.set_title('$t = %1.3f$'%(t[at_index]), fontsize = 15)
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.set_zlim(-Zoom_limit, Zoom_limit)
        error_u = np.linalg.norm(Exact_u[:,:,at_index]-u_pred_reshaped[:,:,at_index],2)/np.linalg.norm(Exact_u[:,:,at_index],2)
        print('L2 relative error: %e' % (error_u))
        ###Observation Probes###
        slice_at_x = 50
        slice_at_y = 50
        ########################
        #######################Slice Along x ##############################
        ax = fig.add_subplot(t_slices,3, 2)
        ax.set_xlabel('x',fontsize = 15)
        ax.set_ylabel('u(x,y,t)',fontsize = 15)
        ax.plot(domain[0],Exact_u[:,slice_at_y,at_index],'r-')
        ax.plot(domain[0],u_pred_reshaped[slice_at_y,:,at_index],'b--')
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(-Zoom_limit, Zoom_limit)
        #######################Slice Along y ##############################
        ax = fig.add_subplot(t_slices,3, 3)
        ax.set_xlabel('y',fontsize = 15)
        ax.set_ylabel('u(x,y,t)')
        ax.plot(domain[1],Exact_u[slice_at_x,:,at_index],'r-', label = 'Exact')
        ax.plot(domain[1],u_pred_reshaped[:,slice_at_x, at_index],'b--', label = 'Prediction')
        ax.set_xlim(lb[1], ub[1])
        ax.set_ylim(-Zoom_limit, Zoom_limit)
        ax.legend()
        plt.show()
    


def plot_weights(model, scale = 1):
    plt.scatter(model.domain.X_f[:,1], model.domain.X_f[:,0], c = model.lambdas[0].numpy(), s = model.lambdas[0].numpy()/float(scale))
    plt.xlabel(model.domain.domain_ids[1])
    plt.ylabel(model.domain.domain_ids[0])
    plt.show()

def plot_glam_values(model, scale = 1):
    plt.scatter(model.t_f, model.x_f, c = model.g(model.col_weights).numpy(), s = model.g(model.col_weights).numpy()/float(scale))
    plt.show()

def plot_residuals(FU_pred, extent):
    fig, ax = plt.subplots()
    ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
                extent=extent,
                origin='lower', aspect='auto')

    #ax.add_collection(ec)
    ax.autoscale_view()
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    cbar = plt.colorbar(ec)
    cbar.set_label('\overline{f}_u prediction')
    plt.show()

def get_griddata(grid, data, dims):
    return griddata(grid, data, dims, method='cubic')
