import java.awt.Dimension;
import java.awt.FileDialog;
import java.awt.Font;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;

@SuppressWarnings("serial")
public class LogoDetector extends Frame {
	public String path = null;
	public String file = null;
	JLabel label = new JLabel(new ImageIcon());
	JLabel label2 = new JLabel(new ImageIcon());
   
	public static void main(String[] args) {
		new LogoDetector("�ΰ� ������");
	}
	
	public LogoDetector(String title) {
		JFrame f = new JFrame(title);
		f.setLocation(100,100);
		f.setPreferredSize(new Dimension(927,700));
		f.setLayout(null);
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);    

		addBtn1(f);
		addBtn2(f);
		addBtn3(f);
      
		JLabel la = new JLabel();
		la.setHorizontalAlignment(JLabel.CENTER); //JLabel ��� ����
		la.setText("LOGO DETECTOR");
		la.setFont(new Font("���� ���",Font.BOLD, 70));
		la.setBounds(100, 450, 727, 100);
		f.add(la);
      
		f.pack();
		f.setVisible(true);
	}
   
   
	public void addBtn1(JFrame f) {
      
		JButton btn1 = new JButton("�̹��� �ҷ�����");
		btn1.setBounds(70, 600, 200, 50);
		btn1.setFont(new Font("���� ���",Font.BOLD, 20));
		f.add(btn1);
		f.setVisible(true);
      
		ActionListener listener1 = new ActionListener() {
         
			@SuppressWarnings("unused")
			@Override
         	public void actionPerformed(ActionEvent e) {
        	 	// TODO Auto-generated method stub
            
        	 	label.setIcon(null);
     
        	 	FileDialog dialog = new FileDialog(f, "�ҷ�����", FileDialog.LOAD);
        	 	dialog.setVisible(true);
            
        	 	path = dialog.getDirectory() + dialog.getFile();
        	 	file = dialog.getFile();
        	 	if( dialog.getFile() == null ) return;
      
	            System.out.println(path);
	            String imgOriginalPath= path;           // ���� �̹��� ���ϸ�
	            String imgTargetPath= path;    // �� �̹��� ���ϸ�
	            String imgFormat = "jpg";                             // �� �̹��� ����. jpg, gif ��
	            int newWidth = 416;                                  // ���� �� ����
	            int newHeight = 416;                                 // ���� �� ����
	            String mainPosition = "W";                             // W:�����߽�, H:�����߽�, X:������ ��ġ��(��������)
       
	            Image image = null;
	            int imageWidth;
	            int imageHeight;
	            double ratio;
	            int w;
	            int h;
            
	            try{
	               // ���� �̹��� ��������
	               image = ImageIO.read(new File(imgOriginalPath));
	       
	               // ���� �̹��� ������ ��������
	               imageWidth = image.getWidth(null);
	               imageHeight = image.getHeight(null);
	       
	               if(mainPosition.equals("W")){    // ���̱���
	       
	                  ratio = (double)newWidth/(double)imageWidth;
	                  w = (int)(imageWidth * ratio);
	                  h = (int)(imageHeight * ratio);
	       
	               }else if(mainPosition.equals("H")){ // ���̱���
	       
	                  ratio = (double)newHeight/(double)imageHeight;
	                  w = (int)(imageWidth * ratio);
	                  h = (int)(imageHeight * ratio);
	       
	               }else{ //������ (��������)
	       
	                  w = newWidth;
	                  h = newHeight;
	               }
	
	               Image resizeImage = image.getScaledInstance(w, h, Image.SCALE_SMOOTH);
	       
	               // �� �̹���  �����ϱ�
	               
	               BufferedImage newImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
	               Graphics g = newImage.getGraphics();
	               g.drawImage(resizeImage, 0, 0, null);
	               g.dispose();
	               ImageIO.write(newImage, imgFormat, new File(imgTargetPath));
	               
	       
	            }catch (Exception ee){
	       
	               ee.printStackTrace();
       
	            }
	            
	            try {
		               
		               File newImage = new File(path);
		               image = ImageIO.read(newImage);
		               
		            } catch (IOException er) {
		               er.printStackTrace();
		            }
	            
	            
	            try {
	                Thread.sleep(1000);
	              } catch (InterruptedException ekl) { }
	
	            String fname = null;
	            
	            try {
	               StringBuffer p = new StringBuffer();
	               p.append(path);
	               p.append(" C:\\Users\\�赵��\\Desktop\\Detection\\data\\face\\test\\images");
	               Process process = Runtime.getRuntime().exec("cmd /c copy " + p);
	               
	               System.out.println(path);
	               
	               path = path.substring(0, path.length()-4);
	               fname = path + ".anno";
	               BufferedWriter fw = new BufferedWriter(new FileWriter(fname, true));
	               fw.write("{\n\t\"puma\": [\n\t\t[\n\t\t\t0, 0, 1, 1 \n\t\t]\n\t]\n}");
	               fw.close(); 
	               StringBuffer p2 = new StringBuffer();
	               p2.append(fname);
	               p2.append(" C:\\Users\\�赵��\\Desktop\\Detection\\data\\face\\test\\annotations");
	               System.out.println(p2);
	               process = Runtime.getRuntime().exec("cmd /c move " + p2);
	               
	            }catch(Exception ex) {
	               ex.printStackTrace();
	               System.exit(1);
	            }

            
	            
	            
	            label = new JLabel(new ImageIcon(image));
	            label.setBounds(10, 10, 400, 400);
	            f.add(label);
	            f.repaint();
	            
	         }
         
      	};
      	btn1.addActionListener(listener1);
   	}
   
	public void addBtn2(JFrame f) {
      
		JButton btn2 = new JButton("�ΰ� ����");
		btn2.setBounds(354, 600, 200, 50);
		btn2.setFont(new Font("���� ���",Font.BOLD, 20));
		f.add(btn2);
		f.repaint();
		f.setVisible(true);
      
		ActionListener listener2 = new ActionListener() {
    	  
			@Override
			public void actionPerformed(ActionEvent e) {
            // TODO Auto-generated method stub
        	 
				try {
        		 
        	 		Process p1 = Runtime.getRuntime( ).exec( 
        	 				"cmd /c cd C:\\Users\\�赵��\\Desktop\\Detection &&" + "python test.py" );      		 
        	 		p1.getErrorStream().close(); 
        	 		p1.getInputStream().close();
        	 		p1.getOutputStream().close(); 
        	 		p1.waitFor();
        	 		p1.destroy();

        	 		}catch(Exception ex) {
        	 			ex.printStackTrace();
        	 			System.exit(1);
        	 		}
         	}
      };
      btn2.addActionListener(listener2);
   }
   
	public void addBtn3(JFrame f) {
	   
		JButton btn3 = new JButton("���� ���");
		btn3.setBounds(637, 600, 200, 50);
		btn3.setFont(new Font("���� ���",Font.BOLD, 20));
		f.add(btn3);
		f.repaint();
		f.setVisible(true);
	   
		ActionListener listener3 = new ActionListener() {
	         
			@SuppressWarnings("unused")
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				Image image = null;
				label2.setIcon(null);
     
				try {
					String pp = "C:\\Users\\�赵��\\Desktop\\Detection\\data\\face\\test\\redraws\\images\\";
					StringBuffer p = new StringBuffer();
					p.append(pp);
					p.append(file);
					System.out.println(p.toString());
					
					File sourceimage = new File(p.toString());
					image = ImageIO.read(sourceimage);
					
				} catch (IOException er) {
					er.printStackTrace();
				}
				
				label2 = new JLabel(new ImageIcon(image));
				label2.setBounds(500, 10, 400, 400);   
				f.add(label2);
				f.repaint();
			}
		};
		btn3.addActionListener(listener3);
	}
   
}