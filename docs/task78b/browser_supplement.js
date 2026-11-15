async page => {
 const c=__TASK78B_CREDENTIALS__;const base='http://127.0.0.1:18072';const checks=[];
 function check(name,ok,detail){checks.push({name,pass:!!ok,detail});if(!ok)throw Error(name);}
 try {
  await page.setViewportSize({width:1440,height:900});await page.goto(base+'/cinema/login/');
  await page.getByLabel('账号',{exact:true}).fill(c.owner);await page.getByLabel('密码',{exact:true}).fill(c.password);await page.getByRole('button',{name:'登录',exact:true}).click();
  for(const width of [900,1024]){
   await page.setViewportSize({width,height:768});await page.getByRole('navigation').getByRole('link',{name:'首页',exact:true}).click();
   const v=await page.evaluate(()=>{const box=s=>{const b=document.querySelector(s).getBoundingClientRect();return {left:b.left,right:b.right,top:b.top,bottom:b.bottom};};return {brand:box('.brand'),nav:box('nav'),account:box('.account'),w:innerWidth,scroll:document.documentElement.scrollWidth};});
   check('medium_nav_no_overlap_'+width,v.brand.right<=v.nav.left&&v.nav.right<=v.account.left&&v.scroll<=width,v);
  }
  await page.setViewportSize({width:1440,height:900});await page.getByRole('navigation').getByRole('link',{name:'我的推荐',exact:true}).click();
  await page.locator('main a[href^="/cinema/personal/traces/"]').first().click();
  const traceURL=page.url();const rows=await page.locator('article.film-card').evaluateAll(cards=>cards.map(c=>({rank:Number(c.querySelector('.card-rank').textContent),title:c.querySelector('h3').textContent,href:c.querySelector('.poster-wrap').getAttribute('href'),score:c.querySelector('.rule-score').textContent})));
  check('twenty_bound_ranked_results',rows.length===20&&rows.every((r,i)=>r.rank===i+1));
  await page.screenshot({path:'/home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/04_recommend_top20.png',fullPage:true});
  await page.setViewportSize({width:375,height:667});await page.goto(base+'/cinema/');await page.getByRole('button',{name:/菜单/}).click();await page.getByRole('navigation').getByRole('link',{name:'影片库',exact:true}).focus();await page.keyboard.press('Escape');
  check('mobile_menu_escape_returns_focus',await page.getByRole('button',{name:/菜单/}).getAttribute('aria-expanded')==='false'&&await page.getByRole('button',{name:/菜单/}).evaluate(e=>e===document.activeElement));
  await page.getByRole('button',{name:/菜单/}).click();await page.getByRole('button',{name:'退出',exact:true}).click();
  const registration=[];
  for(const width of [1440,1366,390,375]){await page.setViewportSize({width,height:width>1000?768:667});await page.goto(base+'/cinema/register/');await page.getByRole('button',{name:'注册',exact:true}).scrollIntoViewIfNeeded();const v=await page.evaluate(()=>({w:innerWidth,h:innerHeight,scroll:document.documentElement.scrollWidth,button:document.querySelector('button[type=submit]').getBoundingClientRect().toJSON()}));registration.push({width,...v});check('register_scroll_button_accessible_'+width,v.scroll<=width&&v.button.top>=0&&v.button.bottom<=v.h,v);}
  return {completed:true,checks,traceURL,rows,registration};
 }catch(error){return {completed:false,failure:String(error),checks};}
}
