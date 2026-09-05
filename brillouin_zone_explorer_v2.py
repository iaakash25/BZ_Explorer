
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.spatial import Voronoi

st.set_page_config(page_title='Brillouin Zone Explorer 2.0', layout='wide')
st.title('Brillouin Zone Explorer 2.0')
st.caption('Real lattice + electron wavelength + wavevector + automatically generated Brillouin zone')

with st.sidebar:
    lattice = st.selectbox('Lattice type', ['1D chain','Square','Rectangular','Triangular','General oblique'])
    a = st.slider('a (Å)', 2.0, 8.0, 4.0, 0.1)

    if lattice == '1D chain':
        b, gamma_deg = None, None
    elif lattice == 'Square':
        b, gamma_deg = a, 90.0
    elif lattice == 'Rectangular':
        b = st.slider('b (Å)', 2.0, 8.0, 5.0, 0.1)
        gamma_deg = 90.0
    elif lattice == 'Triangular':
        b, gamma_deg = a, 60.0
    else:
        b = st.slider('b (Å)', 2.0, 8.0, 5.0, 0.1)
        gamma_deg = st.slider('γ (degrees)', 35.0, 145.0, 75.0, 1.0)

    lam_ratio = st.slider('Wavelength λ/a', 0.5, 8.0, 4.0, 0.1)
    theta_deg = st.slider('Wavevector direction θ (degrees)', 0, 360, 0, 5, disabled=(lattice=='1D chain'))
    show_bisectors = st.checkbox('Show Wigner–Seitz construction', False)

def make_points(v1, v2, n=5):
    pts, labels = [], []
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            pts.append(i*v1+j*v2)
            labels.append((i,j))
    return np.array(pts), labels

def fold_k(kvec, recip_pts):
    best = None
    best_G = None
    best_norm = 1e99
    for G in recip_pts:
        cand = kvec - G
        n = np.linalg.norm(cand)
        if n < best_norm:
            best_norm = n
            best = cand
            best_G = G
    return best, best_G

lam = lam_ratio*a
kmag = 2*np.pi/lam

if lattice == '1D chain':
    G = 2*np.pi/a
    kbz = np.pi/a
    k = kmag
    kfold = ((k + kbz) % G) - kbz

    c1,c2 = st.columns(2)
    with c1:
        fig,ax = plt.subplots(figsize=(7,4))
        x = np.linspace(0,8*a,1000)
        y = 0.55*np.sin(2*np.pi*x/lam)
        atoms = np.arange(0,8.01*a,a)
        ax.scatter(atoms,np.zeros_like(atoms),s=55,zorder=3)
        ax.plot(x,y,linewidth=2)
        ax.set_title('Real-space lattice + electron wave')
        ax.set_xlabel('x (Å)')
        ax.set_yticks([])
        ax.grid(alpha=0.2)
        st.pyplot(fig,use_container_width=True)

    with c2:
        fig,ax = plt.subplots(figsize=(7,4))
        ax.axvspan(-kbz,kbz,alpha=0.12)
        ax.axvline(-kbz,linestyle='--')
        ax.axvline(kbz,linestyle='--')
        ax.scatter([k],[0.2],s=90,label='extended k')
        ax.scatter([kfold],[-0.2],s=90,label='folded k')
        ax.text(-kbz,0.38,'-π/a',ha='center')
        ax.text(kbz,0.38,'+π/a',ha='center')
        ax.text(0,0.38,'Γ',ha='center')
        ax.set_xlim(-max(2.2*kbz,1.25*abs(k)),max(2.2*kbz,1.25*abs(k)))
        ax.set_ylim(-0.55,0.55)
        ax.set_yticks([])
        ax.set_xlabel('k (Å⁻¹)')
        ax.set_title('1D reciprocal space + first BZ')
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig,use_container_width=True)

    st.metric('BZ boundary π/a',f'{kbz:.3f} Å⁻¹')
    st.metric('|k|=2π/λ',f'{k:.3f} Å⁻¹')
    st.stop()

gamma = np.deg2rad(gamma_deg)
a1 = np.array([a,0.0])
a2 = np.array([b*np.cos(gamma),b*np.sin(gamma)])
A = np.column_stack((a1,a2))
B = 2*np.pi*np.linalg.inv(A).T
b1,b2 = B[:,0],B[:,1]

real_pts,_ = make_points(a1,a2,4)
recip_pts,labels = make_points(b1,b2,6)
origin_idx = labels.index((0,0))

vor = Voronoi(recip_pts)
region = vor.regions[vor.point_region[origin_idx]]
if -1 in region or not region:
    st.error('Could not construct first BZ for this geometry.')
    st.stop()

bz = vor.vertices[region]
ctr = bz.mean(axis=0)
order = np.argsort(np.arctan2(bz[:,1]-ctr[1],bz[:,0]-ctr[0]))
bz = bz[order]

d = np.linalg.norm(recip_pts,axis=1)
dmin = d[d>1e-10].min()
nearest = recip_pts[np.isclose(d,dmin,rtol=1e-6,atol=1e-8)]

theta = np.deg2rad(theta_deg)
k_ext = kmag*np.array([np.cos(theta),np.sin(theta)])
k_fold,Gused = fold_k(k_ext,recip_pts)

def plot_real():
    fig,ax = plt.subplots(figsize=(6.5,6))
    ax.scatter(real_pts[:,0],real_pts[:,1],s=25)
    cell = np.array([[0,0],a1,a1+a2,a2,[0,0]])
    ax.plot(cell[:,0],cell[:,1],linewidth=2)

    dvec = np.array([np.cos(theta),np.sin(theta)])
    pvec = np.array([-np.sin(theta),np.cos(theta)])
    s = np.linspace(-5*a,5*a,900)
    wave = 0.35*a*np.sin(2*np.pi*s/lam)
    xy = np.outer(s,dvec)+np.outer(wave,pvec)
    ax.plot(xy[:,0],xy[:,1],linewidth=2)

    ax.set_aspect('equal')
    ax.set_title('Real lattice + electron wave')
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.grid(alpha=0.2)
    lim = 4.2*max(np.linalg.norm(a1),np.linalg.norm(a2))
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    return fig

def plot_bz():
    fig,ax = plt.subplots(figsize=(6.5,6))
    ax.scatter(recip_pts[:,0],recip_pts[:,1],s=16)
    closed = np.vstack([bz,bz[0]])
    ax.plot(closed[:,0],closed[:,1],linewidth=2.5)
    ax.fill(bz[:,0],bz[:,1],alpha=0.12)
    ax.scatter([0],[0],s=70)
    ax.text(0.04,0.04,'Γ',fontsize=12)

    ax.scatter(nearest[:,0],nearest[:,1],s=48)
    for G in nearest:
        ax.plot([0,G[0]],[0,G[1]],'--',linewidth=0.9,alpha=0.7)
        if show_bisectors:
            rhs=np.dot(G,G)/2
            t=np.linspace(-20,20,400)
            if abs(G[1])>abs(G[0]):
                x=t
                y=(rhs-G[0]*x)/G[1]
            else:
                y=t
                x=(rhs-G[1]*y)/G[0]
            ax.plot(x,y,':',linewidth=1)

    ax.arrow(0,0,k_ext[0],k_ext[1],length_includes_head=True,linewidth=2)
    ax.scatter([k_ext[0]],[k_ext[1]],s=90,label='extended k')
    ax.arrow(0,0,k_fold[0],k_fold[1],length_includes_head=True,linewidth=2)
    ax.scatter([k_fold[0]],[k_fold[1]],s=90,label='folded k')

    lim=max(np.linalg.norm(b1),np.linalg.norm(b2),np.linalg.norm(k_ext),np.max(np.linalg.norm(bz,axis=1)))*1.55
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_aspect('equal')
    ax.set_title('Reciprocal lattice + actual first Brillouin zone')
    ax.set_xlabel('kₓ (Å⁻¹)')
    ax.set_ylabel('kᵧ (Å⁻¹)')
    ax.grid(alpha=0.2)
    ax.legend()
    return fig

c1,c2=st.columns(2)
with c1:
    st.pyplot(plot_real(),use_container_width=True)
with c2:
    st.pyplot(plot_bz(),use_container_width=True)

m1,m2,m3,m4=st.columns(4)
m1.metric('a',f'{a:.3f} Å')
m2.metric('λ',f'{lam:.3f} Å')
m3.metric('|k|',f'{kmag:.3f} Å⁻¹')
m4.metric('|k folded|',f'{np.linalg.norm(k_fold):.3f} Å⁻¹')

st.latex(r'k=\frac{2\pi}{\lambda}')
st.latex(r'\mathbf{a}_i\cdot\mathbf{b}_j=2\pi\delta_{ij}')

if lattice=='Triangular':
    st.info('Triangular real lattice → triangular reciprocal lattice → six nearest reciprocal neighbors → hexagonal first Brillouin zone.')

st.caption('Changing λ moves the k-point. Changing a, b or γ changes the BZ itself.')
